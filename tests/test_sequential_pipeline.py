import json
import tempfile
import unittest
from pathlib import Path

from flashrag.dataset import Dataset
from flashrag.pipeline.pipeline import SequentialPipeline


class FakeRetriever:
    def __init__(self, retrieval_results, doc_scores):
        self.retrieval_results = retrieval_results
        self.doc_scores = doc_scores
        self.calls = []

    def batch_search(self, queries, return_score=False):
        self.calls.append((queries, return_score))
        if return_score:
            return self.retrieval_results, self.doc_scores
        return self.retrieval_results


class FakeGenerator:
    def generate(self, prompts):
        return ["answer" for _ in prompts]


class FakePromptTemplate:
    def get_string(self, question, retrieval_result):
        return f"{question}: {retrieval_result[0]['contents']}"


class TestSequentialPipeline(unittest.TestCase):
    def test_run_saves_retrieval_scores_in_intermediate_data(self):
        retrieval_results = [
            [{"id": "doc-1", "contents": "First document"}],
            [{"id": "doc-2", "contents": "Second document"}],
        ]
        doc_scores = [[0.9], [0.7]]
        retriever = FakeRetriever(retrieval_results, doc_scores)
        pipeline = SequentialPipeline.__new__(SequentialPipeline)
        pipeline.retriever = retriever
        pipeline.generator = FakeGenerator()
        pipeline.prompt_template = FakePromptTemplate()
        pipeline.refiner = None
        pipeline.use_fid = False
        pipeline.evaluate = lambda dataset, do_eval=True, pred_process_fun=None: dataset
        dataset = Dataset(
            config={"dataset_name": "test"},
            data=[
                {"id": "1", "question": "Question one?"},
                {"id": "2", "question": "Question two?"},
            ],
        )

        result = pipeline.run(dataset, do_eval=False)

        self.assertEqual(retriever.calls, [(["Question one?", "Question two?"], True)])
        self.assertEqual(result.retrieval_result, retrieval_results)
        self.assertEqual(result.doc_scores, doc_scores)

        with tempfile.TemporaryDirectory() as directory:
            save_path = Path(directory) / "intermediate_data.json"
            result.save(save_path)
            saved_data = json.loads(save_path.read_text())

        self.assertEqual(
            [item["output"]["doc_scores"] for item in saved_data],
            doc_scores,
        )


if __name__ == "__main__":
    unittest.main()
