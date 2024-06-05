from typing import cast, List
import json
from collections import Counter
import numpy as np
import torch
import faiss
from flashrag.retriever.utils import load_model, pooling

class BaseJudger:
    """Base object of Judger, used for judging whether to retrieve"""

    def __init__(self, config):
        self.config = config
        self.name = config['judger_name']
        self.device = config['device']

    def run(self, item) -> str:
        """Get judgement result.

        Args:
            item: dataset item, contains question, retrieval result...

        Returns:
            judgement: bool, whether to retreive
        """
        pass

    def batch_run(self, dataset, batch_size = None) -> List[str]:
        return [self.run(item) for item in dataset]

class SKRJudger(BaseJudger):
    """Implementation for SKR-knn
    Paper link: https://aclanthology.org/2023.findings-emnlp.691.pdf
    """

    def __init__(self, config):
        super().__init__(config)
        self.model_path = config['judger_model_path']
        self.training_data_path = config['judger_training_data_path']
        self.encoder, self.tokenizer = load_model(model_path = self.model_path,
                                                  use_fp16 = False)
        self.topk = config['judger_topk'] if 'judger_topk' in config else 5
        self.batch_size = config['judger_batch_size'] if 'judger_batch_size' in config else 64
        self.max_length = config['judger_max_length'] if 'judger_max_length' in config else 128

        with open(self.training_data_path, "r") as f:
            self.training_data = json.load(f)
        # count number of pos & neg samples in training data
        self.training_data_counter = Counter([item['judgement'].strip() for item in self.training_data])
        self.training_pos_num = self.training_data_counter['ir_better']
        self.training_neg_num = self.training_data_counter['ir_worse']
        self.training_data_num = sum(self.training_data_counter.values())

        # encode training question into faiss
        training_questions = [item['question'] for item in self.training_data]
        all_embeddings = self.encode(training_questions)
        faiss_index = faiss.index_factory(all_embeddings.shape[-1], 'Flat', faiss.METRIC_L2)
        faiss_index.add(all_embeddings)
        self.faiss = faiss_index



    def encode(self, contents:list):
        inputs = self.tokenizer(
                    contents,
                    padding=True,
                    truncation=True,
                    return_tensors='pt',
                    max_length=self.max_length,
        ).to('cuda')
        with torch.no_grad():
            output = self.encoder(**inputs, return_dict=True)
        embeddings = pooling(output.pooler_output,
                                output.last_hidden_state,
                                inputs['attention_mask'],
                                'pooler')

        embeddings = cast(torch.Tensor, embeddings)
        embeddings = torch.nn.functional.normalize(embeddings, dim=-1).detach()

        all_embeddings = embeddings.cpu().numpy()
        #all_embeddings = np.concatenate(all_embeddings, axis=0)
        all_embeddings = all_embeddings.astype(np.float32)

        return all_embeddings

    def judge(self, dataset):
        questions = dataset.question

        all_judgements = []
        for start_idx in range(0,len(questions), self.batch_size):
            batch_question = questions[start_idx:start_idx+self.batch_size]
            batch_emb = self.encode(batch_question)
            scores, batch_idxs = self.faiss.search(batch_emb, k=self.topk)

            for idxs in batch_idxs:
                topk_samples = [self.training_data[idx]['judgement'].strip() for idx in idxs]
                topk_counter = Counter(topk_samples)

                # count number of pos & neg samples in topk
                ir_better_num = topk_counter['ir_better']
                ir_worse_num = topk_counter['ir_worse']
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



