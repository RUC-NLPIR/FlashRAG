from typing import List
import torch
import warnings
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from flashrag.retriever.encoder import Encoder


class BaseReranker:
    r"""Base object for all rerankers."""

    def __init__(self, config):
        self.config = config
        self.reranker_model_name = config["rerank_model_name"]
        self.reranker_model_path = config["rerank_model_path"]
        self.topk = config["rerank_topk"]
        self.max_length = config["rerank_max_length"]
        self.batch_size = config["rerank_batch_size"]
        self.device = config["device"]

    def get_rerank_scores(self, query_list: List[str], doc_list: List[str], batch_size):
        """Return flatten list of scores for each (query,doc) pair
        Args:
            query_list: List of N queries
            doc_list:  Nested list of length N, each element corresponds to K documents of a query

        Return:
            [score(q1,d1), score(q1,d2),... score(q2,d1),...]
        """
        all_scores = []
        return all_scores

    @torch.inference_mode(mode=True)
    def rerank(self, query_list, doc_list, batch_size=None, topk=None):
        r"""Rerank doc_list."""
        if batch_size is None:
            batch_size = self.batch_size
        if topk is None:
            topk = self.topk
        if isinstance(query_list, str):
            query_list = [query_list]
        if not isinstance(doc_list[0], list):
            doc_list = [doc_list]

        assert len(query_list) == len(doc_list)
        if topk < min([len(docs) for docs in doc_list]):
            warnings.warn("The number of doc returned by the retriever is less than the topk.")

        # get doc contents
        doc_contents = []
        for docs in doc_list:
            if all([isinstance(doc, str) for doc in docs]):
                doc_contents.append([doc for doc in docs])
            else:
                doc_contents.append([doc["contents"] for doc in docs])

        all_scores = self.get_rerank_scores(query_list, doc_contents, batch_size)
        assert len(all_scores) == sum([len(docs) for docs in doc_list])

        # sort docs
        start_idx = 0
        final_scores = []
        final_docs = []
        for docs in doc_list:
            doc_scores = all_scores[start_idx : start_idx + len(docs)]
            doc_scores = [float(score) for score in doc_scores]
            sort_idxs = np.argsort(doc_scores)[::-1][:topk]
            start_idx += len(docs)

            final_docs.append([docs[idx] for idx in sort_idxs])
            final_scores.append([doc_scores[idx] for idx in sort_idxs])

        return final_docs, final_scores


class CrossReranker(BaseReranker):
    def __init__(self, config):
        super().__init__(config)
        self.tokenizer = AutoTokenizer.from_pretrained(self.reranker_model_path)
        self.ranker = AutoModelForSequenceClassification.from_pretrained(self.reranker_model_path, num_labels=1)
        self.ranker.eval()
        self.ranker.to(self.device)

    @torch.inference_mode(mode=True)
    def get_rerank_scores(self, query_list, doc_list, batch_size):
        # flatten all pairs
        all_pairs = []
        for query, docs in zip(query_list, doc_list):
            all_pairs.extend([[query, doc] for doc in docs])
        all_scores = []
        for start_idx in tqdm(range(0, len(all_pairs), batch_size), desc="Reranking process: "):
            pair_batch = all_pairs[start_idx : start_idx + batch_size]

            inputs = self.tokenizer(
                pair_batch, padding=True, truncation=True, return_tensors="pt", max_length=self.max_length
            ).to(self.device)
            batch_scores = (
                self.ranker(**inputs, return_dict=True)
                .logits.view(
                    -1,
                )
                .float()
                .cpu()
            )
            all_scores.extend(batch_scores)

        return all_scores


class BiReranker(BaseReranker):
    def __init__(self, config):
        super().__init__(config)
        self.encoder = Encoder(
            model_name=self.reranker_model_name,
            model_path=self.reranker_model_path,
            pooling_method=config["rerank_pooling_method"],
            max_length=self.max_length,
            use_fp16=config["rerank_use_fp16"],
        )

    def get_rerank_scores(self, query_list, doc_list, batch_size):
        query_emb = []
        for start_idx in range(0, len(query_list), batch_size):
            query_batch = query_list[start_idx : start_idx + batch_size]
            batch_emb = self.encoder.encode(query_batch, is_query=True)
            query_emb.append(batch_emb)
        query_emb = np.concatenate(query_emb, axis=0)

        flat_doc_list = sum(doc_list, [])
        doc_emb = []
        for start_idx in range(0, len(flat_doc_list), batch_size):
            doc_batch = flat_doc_list[start_idx : start_idx + batch_size]
            batch_emb = self.encoder.encode(doc_batch, is_query=False)
            doc_emb.append(batch_emb)
        doc_emb = np.concatenate(doc_emb, axis=0)

        scores = query_emb @ doc_emb.T  # K*L
        all_scores = []
        score_idx = 0
        for idx, doc in enumerate(doc_list):
            all_scores.extend(scores[idx, score_idx : score_idx + len(doc)])
            score_idx += len(doc)

        return all_scores
