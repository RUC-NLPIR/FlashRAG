from typing import cast, List
import json
from tqdm.auto import trange
from collections import Counter
import numpy as np
import torch
import faiss
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from flashrag.retriever.utils import load_model, pooling
from flashrag.dataset import Dataset


class BaseJudger:
    """Base object of Judger, used for judging whether to retrieve"""

    def __init__(self, config):
        self.config = config
        self.name = config["judger_name"] if "judger_name" in config else None
        self.judger_config = config["judger_config"] if "judger_config" in config else {}
        self.device = config["device"]

    def run(self, item) -> str:
        """Get judgement result.

        Args:
            item: dataset item, contains question, retrieval result...

        Returns:
            judgement: bool, whether to retreive
        """
        pass

    def batch_run(self, dataset, batch_size=None) -> List[str]:
        return [self.run(item) for item in dataset]


class SKRJudger(BaseJudger):
    """Implementation for SKR-knn
    Paper link: https://aclanthology.org/2023.findings-emnlp.691.pdf
    """

    def __init__(self, config):
        super().__init__(config)
        self.model_path = self.judger_config["model_path"]
        self.training_data_path = self.judger_config["training_data_path"]
        self.encoder, self.tokenizer = load_model(model_path=self.model_path, use_fp16=False)
        self.topk = self.judger_config["topk"] if "topk" in self.judger_config else 5
        self.batch_size = self.judger_config["batch_size"] if "batch_size" in config else 64
        self.max_length = self.judger_config["max_length"] if "max_length" in config else 128

        with open(self.training_data_path, "r") as f:
            self.training_data = json.load(f)
        # count number of pos & neg samples in training data
        self.training_data_counter = Counter([item["judgement"].strip() for item in self.training_data])
        self.training_pos_num = self.training_data_counter["ir_better"]
        self.training_neg_num = self.training_data_counter["ir_worse"]
        self.training_data_num = sum(self.training_data_counter.values())

        # encode training question into faiss
        training_questions = [item["question"] for item in self.training_data]
        all_embeddings = self.encode(training_questions)
        faiss_index = faiss.index_factory(all_embeddings.shape[-1], "Flat", faiss.METRIC_L2)
        faiss_index.add(all_embeddings)
        self.faiss = faiss_index

    @torch.inference_mode(mode=True)
    def encode(self, contents: list):
        inputs = self.tokenizer(
            contents,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=self.max_length,
        ).to("cuda")
        output = self.encoder(**inputs, return_dict=True)
        embeddings = pooling(output.pooler_output, output.last_hidden_state, inputs["attention_mask"], "pooler")

        embeddings = cast(torch.Tensor, embeddings)
        embeddings = torch.nn.functional.normalize(embeddings, dim=-1).detach()

        all_embeddings = embeddings.cpu().numpy()
        # all_embeddings = np.concatenate(all_embeddings, axis=0)
        all_embeddings = all_embeddings.astype(np.float32)

        return all_embeddings

    def judge(self, dataset):
        if isinstance(dataset, Dataset):
            questions = dataset.question
        elif isinstance(dataset, list):
            questions = dataset
        elif isinstance(dataset, str):
            questions = [dataset]
        else:
            raise TypeError("dataset must be a Dataset or a list of str.")
        
        all_judgements = []
        for start_idx in range(0, len(questions), self.batch_size):
            batch_question = questions[start_idx : start_idx + self.batch_size]
            batch_emb = self.encode(batch_question)
            scores, batch_idxs = self.faiss.search(batch_emb, k=self.topk)

            for idxs in batch_idxs:
                topk_samples = [self.training_data[idx]["judgement"].strip() for idx in idxs]
                topk_counter = Counter(topk_samples)

                # count number of pos & neg samples in topk
                ir_better_num = topk_counter["ir_better"]
                ir_worse_num = topk_counter["ir_worse"]
                topk_delta = ir_better_num - ir_worse_num

                training_data_delta = self.training_pos_num - self.training_neg_num

                # provide judgments based on the formula in the paper
                if training_data_delta < 0:
                    if topk_delta < 0 and topk_delta <= int(training_data_delta * self.topk / self.training_data_num):
                        judgement = False
                    else:
                        judgement = True
                else:
                    if topk_delta > 0 and topk_delta >= int(training_data_delta * self.topk / self.training_data_num):
                        judgement = True
                    else:
                        judgement = False

                all_judgements.append(judgement)

        return all_judgements


class AdaptiveJudger(BaseJudger):
    """Implementation for Adaptive-RAG
    Paper link: https://aclanthology.org/2024.naacl-long.389.pdf
    """

    def __init__(self, config):
        super().__init__(config)
        self.model_path = self.judger_config["model_path"]
        self.batch_size = self.judger_config.get("batch_size", 16)
        self.max_length = self.judger_config.get("max_length", 512)

        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model.eval()
        self.model.cuda()

    @torch.inference_mode(mode=True)
    def judge(self, dataset):
        if isinstance(dataset, Dataset):
            questions = dataset.question
        elif isinstance(dataset, list):
            questions = dataset
        elif isinstance(dataset, str):
            questions = [dataset]
        else:
            raise TypeError("dataset must be a Dataset or a list of str.")
            
        questions = [q.strip() for q in questions]

        all_preds = []
        for idx in trange(0, len(questions), self.batch_size, desc="Judger process: "):
            batch_input = questions[idx : idx + self.batch_size]
            batch_input = self.tokenizer(
                batch_input,
                truncation=True,
                padding=True,
                max_length=512,
                return_tensors="pt",
            ).to(self.model.device)

            scores = self.model.generate(
                **batch_input, return_dict_in_generate=True, output_scores=True, max_length=self.max_length
            ).scores[0]

            probs = (
                torch.nn.functional.softmax(
                    torch.stack(
                        [
                            scores[:, self.tokenizer("A").input_ids[0]],
                            scores[:, self.tokenizer("B").input_ids[0]],
                            scores[:, self.tokenizer("C").input_ids[0]],
                        ]
                    ),
                    dim=0,
                )
                .detach()
                .cpu()
                .numpy()
            )

            preds_labels = np.argmax(probs, 0)
            label_to_option = {
                0: "A",
                1: "B",
                2: "C",
            }
            preds = [label_to_option[pred] for pred in preds_labels]
            all_preds.extend(preds)

        return all_preds
