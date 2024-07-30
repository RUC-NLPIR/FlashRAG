# This file is used for generating LM supervised dataset to finetune retrievers
# Implementation details are learned from REPLUG:https://arxiv.org/abs/2301.12652
import json
import random

import fire
import torch
from tqdm import tqdm

from flashrag.config import Config
from flashrag.prompt import PromptTemplate
from flashrag.utils import get_dataset
from flashrag.utils import get_retriever, get_generator


class LMProbCalculator:
    """
    Clculating the likelihood of the ground truth
    when LM uses every document retrieved from
    top-k retrieved ones in context
    """

    def __init__(self, config):
        # Load your own components
        self.retriever = get_retriever(config)
        self.generator = get_generator(config)
        self.prompt_template = PromptTemplate(config)

    def run(self, dataset):
        input_query = dataset.question
        answers = [answer[0] if len(answer) == 1 else random.choice(answer) for answer in dataset.golden_answers]
        retrieval_results = self.retriever.batch_search(input_query)
        data_ls = []
        N = len(answers)
        for i in tqdm(range(N)):
            q, res_list, answer = dataset.question[i], retrieval_results[i], answers[i]
            docs = []
            scores = []
            for res in res_list:
                input_prompt = self.prompt_template.get_string(question=q, retrieval_result=[res])
                score = self.calculate_prob(input_prompt, answer)
                scores.append(score)
                docs.append(res["contents"])
            scores = torch.softmax(torch.tensor(scores), dim=-1).tolist()
            data_ls.append({"query": q, "pos": docs, "scores": scores})
        return data_ls

    def calculate_prob(self, prompt, answer):
        _, answer_probs = self.generator.cal_gen_probs(prompt, answer)
        return float(answer_probs.mean())


def main(
    dataset_name="nq",  # qa dataset
    split="test",  # split
    num=4000,  # number of query-document pairs
    gpu_id="0",
    output="lmsft.jsonl",  # output path
    topk=20,  # number of retrieved documents
):
    config_dict = {
        "save_note": "replug_lsr",
        "gpu_id": gpu_id,
        "dataset_name": dataset_name,
        "test_sample_num": num,
        "split": ["test", "dev"],
        "retrieval_topk": topk,
        "framework": "hf",
    }
    config = Config("my_config.yaml", config_dict)
    all_split = get_dataset(config)
    test_data = all_split[split]
    lm_prob_calculator = LMProbCalculator(config)
    data_ls = lm_prob_calculator.run(test_data)
    with open(output, "w") as f:
        for data in data_ls:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    fire.Fire(main)
