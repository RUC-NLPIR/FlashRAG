"""A retrieval budget must leave a generation turn to consume the final results."""

import json

import pytest

from flashrag.dataset import Dataset
from flashrag.pipeline import (
    AutoRefinePipeline,
    O2SearcherPipeline,
    ReasoningPipeline,
    SearchR1Pipeline,
    SimpleDeepSearcherPipeline,
)

PIPELINES = [
    (
        ReasoningPipeline,
        "<|begin_of_query|>capital<|end_of_query|>",
        "<answer>Paris</answer>",
    ),
    (SearchR1Pipeline, "<search>capital</search>", "<answer>Paris</answer>"),
    (AutoRefinePipeline, "<search>capital</search>", "<answer>Paris</answer>"),
    (
        SimpleDeepSearcherPipeline,
        "<|begin_search_query|>capital<|end_search_query|>",
        r"\boxed{Paris}",
    ),
    (
        O2SearcherPipeline,
        "<search><query>capital</query></search>",
        "<answer>Paris</answer>",
    ),
]


class ScriptedGenerator:
    def __init__(self):
        self.responses = []
        self.prompts = []

    def generate(self, prompts, **kwargs):
        self.prompts.append(list(prompts))
        response = self.responses.pop(0)
        assert len(response) == len(prompts)
        return response


class RecordingRetriever:
    def __init__(self):
        self.queries = []

    def batch_search(self, queries):
        self.queries.append(list(queries))
        return [[{"contents": "France\nParis is the capital."}] for _ in queries]


class PromptTemplate:
    def get_string(self, question=None, messages=None, **kwargs):
        if messages is not None:
            return "\n".join(message["content"] for message in messages)
        return question

    def truncate_prompt(self, prompt):
        return prompt

    def _get_tokenizer(self):
        return self

    def encode(self, text, **kwargs):
        return [list(text)]

    def decode(self, tokens, **kwargs):
        return "".join(tokens)


@pytest.fixture(params=PIPELINES, ids=lambda case: case[0].__name__)
def setup_pipeline(request):
    pipeline_class, query, answer = request.param
    config = {
        "device": "cpu",
        "dataset_name": "retrieval-budget-test",
        "save_dir": ".",
        "save_metric_score": False,
        "save_intermediate_data": False,
        "save_retrieval_cache": False,
        "metrics": ["em"],
    }
    generator = ScriptedGenerator()
    retriever = RecordingRetriever()
    pipeline = pipeline_class(
        config,
        prompt_template=PromptTemplate(),
        generator=generator,
        retriever=retriever,
    )
    dataset = Dataset(
        config, data=[{"question": "Capital?", "golden_answers": ["Paris"]}]
    )
    return pipeline, dataset, generator, retriever, query, answer


@pytest.mark.parametrize("budget", [0, 1, 2])
def test_answer_after_final_allowed_retrieval(setup_pipeline, budget, tmp_path):
    pipeline, dataset, generator, retriever, query, answer = setup_pipeline
    pipeline.max_retrieval_num = budget
    generator.responses = [[query] for _ in range(budget)] + [[answer]]

    result = pipeline.run(dataset)

    assert result.pred == ["Paris"]
    assert result[0].finish_reason == "Finished"
    assert result[0].retrieved_times == budget
    assert len(retriever.queries) == budget
    assert len(generator.prompts) == budget + 1
    assert not generator.responses
    assert result[0].metric_score["em"] == 1
    if budget:
        assert "Paris is the capital." in generator.prompts[-1][0]
        assert len(result[0].retrieval_results) == budget
    path = tmp_path / "results.json"
    result.save(path)
    assert json.loads(path.read_text())[0]["output"]["pred"] == "Paris"


@pytest.mark.parametrize("budget", [0, 1])
def test_extra_search_is_rejected_without_exceeding_budget(setup_pipeline, budget):
    pipeline, dataset, generator, retriever, query, _ = setup_pipeline
    pipeline.max_retrieval_num = budget
    generator.responses = [[query] for _ in range(budget + 1)]

    pipeline.run(dataset)

    assert dataset[0].finish_flag
    assert dataset[0].finish_reason == "Reach max retrieval number"
    assert dataset.pred == ["No valid answer found"]
    assert dataset[0].retrieved_times == budget
    assert len(retriever.queries) == budget
    assert len(generator.prompts) == budget + 1
    assert not generator.responses


def test_mixed_batch_preserves_finished_answer_at_limit(setup_pipeline):
    pipeline, _, generator, retriever, query, answer = setup_pipeline
    dataset = Dataset(
        pipeline.config,
        data=[
            {"question": question, "golden_answers": ["Paris"]}
            for question in ["Early?", "Final?", "More?"]
        ],
    )
    pipeline.max_retrieval_num = 1
    generator.responses = [[answer, query, query], [answer, query]]

    pipeline.run(dataset)

    assert dataset.pred == ["Paris", "Paris", "No valid answer found"]
    assert dataset.finish_flag == [True, True, True]
    assert dataset.retrieved_times == [0, 1, 1]
    assert [len(batch) for batch in generator.prompts] == [3, 2]
    assert retriever.queries == [["capital", "capital"]]
    assert not generator.responses


def test_empty_dataset_does_not_generate_or_retrieve(setup_pipeline):
    pipeline, _, generator, retriever, _, _ = setup_pipeline
    dataset = Dataset(pipeline.config, data=[])

    assert pipeline.run(dataset, do_eval=False) is dataset
    assert not generator.prompts
    assert not retriever.queries


def test_unstructured_final_response_terminates(setup_pipeline):
    pipeline, dataset, generator, retriever, query, _ = setup_pipeline
    pipeline.max_retrieval_num = 1
    generator.responses = [[query], ["Paris"]]

    pipeline.run(dataset)

    assert dataset[0].finish_flag
    assert dataset.pred == (
        ["No valid answer found"]
        if isinstance(pipeline, O2SearcherPipeline)
        else ["Paris"]
    )
    assert len(generator.prompts) == 2
    assert len(retriever.queries) == 1
    assert not generator.responses


@pytest.mark.parametrize(
    "setup_pipeline", [PIPELINES[-1]], indirect=True, ids=["O2SearcherPipeline"]
)
def test_multiple_queries_use_one_retrieval_round(setup_pipeline):
    pipeline, dataset, generator, retriever, _, answer = setup_pipeline
    pipeline.max_retrieval_num = 1
    generator.responses = [
        ["<search><query>capital</query><query>France</query></search>"],
        [answer],
    ]

    pipeline.run(dataset)

    assert dataset.pred == ["Paris"]
    assert retriever.queries == [["capital", "France"]]
    assert dataset[0].retrieved_times == 1
    assert len(dataset[0].retrieval_results[0]["docs"]) == 2
    assert len(generator.prompts) == 2
