from typing import cast, List, Union, Tuple
from transformers import AutoModelForSeq2SeqLM,AutoTokenizer
from flashrag.retriever.utils import load_model, pooling
from tqdm import tqdm
import re
import torch
import json
from collections import Counter
import numpy as np
import faiss

class BaseJudger:
    r"""Base object of Judger, used for judging whether to retrieve"""

    def __init__(self, config):
        self.config = config
        self.name = config['judger_name']
        self.device = config['device']
    
    def run(self, item) -> str:
        r"""Get judgement result.

        Args:
            item: dataset item, contains question, retrieval result...

        Returns:
            judgement: bool, whether to retreive
        """
        pass

    def batch_run(self, dataset, batch_size = None) -> List[str]:
        return [self.run(item) for item in dataset]

class SKRJudger(BaseJudger):
    """Implementation for SKR-knn"""
    def __init__(self, config):
        super().__init__(config)
        self.model_path = config['judger_model_path']
        self.training_data_path = config['judger_training_data_path']
        self.encoder, self.tokenizer = load_model(model_path = self.model_path, 
                                                  use_fp16 = False)
        self.topk = config['judger_topk']
        self.batch_size = config['judger_batch_size'] if 'judger_batch_size' in config else 64
        self.max_length = config['judger_max_length'] if 'judger_max_length' in config else 128

        with open(self.training_data_path, "r") as f:
            self.training_data = json.load(f)
        # count number of pos & neg samples in training data
        training_data_counter = Counter([item['judgement'].strip() for item in self.training_data])
        self.training_pos_num = training_data_counter['ir_better']
        self.training_neg_num = training_data_counter['ir_worse']

        # encode training question into faiss
        training_questions = [item['question'] for item in self.training_data]
        all_embeddings = self.encode(training_questions)
        faiss_index = faiss.index_factory(all_embeddings.shape[-1], 'Flat', faiss.METRIC_L2)
        faiss_index.add(all_embeddings)
        self.faiss = faiss_index
        

    
    def encode(self, contents:list):
        all_embeddings = []
        for start_index in tqdm(range(0, len(contents), self.batch_size), 
                                desc="Encoding data: ",
                                disable=len(contents) < self.batch_size):
            sentences_batch = contents[start_index:start_index + self.batch_size]
            inputs = self.tokenizer(
                        sentences_batch,
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

            embeddings = embeddings.cpu().numpy()
            all_embeddings.append(embeddings)


        all_embeddings = np.concatenate(all_embeddings, axis=0)
        all_embeddings = all_embeddings.astype(np.float32)

        return all_embeddings

    def judge(self, dataset):
        questions = dataset.question
        all_embeddings = self.encode(questions)

        all_judgements = []
        for q_emb in all_embeddings:
            # search topk nearest training sample
            scores, idxs = self.faiss.search(q_emb, k=self.topk)
            idxs = idxs[0]
            topk_samples = [self.training_data[idx]['judgement'].strip() for idx in idxs]
            topk_counter = Counter(topk_samples)

            # count number of pos & neg samples in topk
            ir_better_num = topk_counter['ir_better']
            ir_worse_num = topk_counter['ir_worse']
            topk_delta = ir_better_num - ir_worse_num

            training_data_delta = self.training_pos_num - self.training_pos_num

            # provide judgments based on the formula in the paper
            if training_data_delta < 0:
                if topk_delta < 0 and topk_delta <= int(training_data_delta * self.topk / sum(self.training_data)):
                    judgement = False
                else:
                    judgement = True
            else:
                if topk_delta > 0 and topk_delta >= int(training_data_delta * self.topk / sum(self.training_data)):
                    judgement = True
                else:
                    judgement = False

            all_judgements.append(judgement)

        return all_judgements
            


