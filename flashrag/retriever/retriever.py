import faiss
import json
from abc import ABC, abstractmethod
from typing import List, Dict
import numpy as np
import torch

from pyserini.search.lucene import LuceneSearcher
from pyserini.index.lucene import IndexReader

from torch import Tensor
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModel
from flashrag.retriever.utils import load_model, pooling, base_content_function, load_database

        
class BaseRetriever(ABC):
    r"""Base object for all retrievers."""

    def __init__(self, config):
        self.config = config
        self.retrieval_method = config['retrieval_method']
        self.topk = config['retrieval_topk']

        self.index_path = config['index_path']
        self.corpus_database_path = config['corpus_database_path']


    @abstractmethod
    def search(self, query: str, num: int) -> List[Dict[str, str]]:
        r"""Retrieve topk relevant documents in corpus.
        
        Return:
            list: contains information related to the document, including:
                contents: used for building index
                title: (if provided)
                text: (if provided)

        """

        pass

    
class BM25Retriever(BaseRetriever):
    r"""BM25 retriever based on pre-built pyserini index."""

    def __init__(self, config):
        super().__init__(config)
        self.searcher = LuceneSearcher(self.index_path)
        self.contain_doc = self._check_contain_doc()
        if not self.contain_doc:
            self.corpus = load_database(self.corpus_database_path)
        
    def _check_contain_doc(self):
        r"""Check if the index contains document content
        """
        return self.searcher.doc(0).raw() is not None

    def search(self, query: str, num: int = None) -> List[Dict[str, str]]:
        if num is None:
            num = self.topk
        hits = self.searcher.search(query, num)
        # TODO: Supplement the situation when there are not enough results recalled
        if self.contain_doc:
            all_contents = [json.loads(self.searcher.doc(hits[i].docid).raw())['contents'] for i in range(num)]
            results = [{'title': content.split("\n")[0].strip("\""), 
                        'text': "\n".join(content.split("\n")[1:]),
                        'contents': content} for content in all_contents]
        else:
            results = [self.corpus.get(hits[i].docid) for i in range(num)]

        return results


class DenseRetriever(BaseRetriever):
    r"""Dense retriever based on pre-built faiss index."""

    def __init__(self, config: dict):
        super().__init__(config)
        # TODO: adapt to pyserini index
        self.index = faiss.read_index(self.index_path)
        self.corpus = load_database(self.corpus_database_path)
        self.encoder, self.tokenizer = load_model(model_path = config['retrieval_model_path'], 
                                                  use_fp16 = config['retrieval_use_fp16'])
        self.topk = config['retrieval_topk']
        self.pooling_method = self.config['retrieval_pooling_method'] 
        self.query_max_length = self.config['retrieval_query_max_length']

    
    @torch.no_grad()
    def _encode(self, query: str) -> np.ndarray:
        # processing query for different encoders
        if self.retrieval_method == "e5":
            query = f"query: {query}"

        inputs = self.tokenizer(query, 
                                max_length = self.query_max_length, 
                                padding = True, 
                                truncation = True, 
                                return_tensors = "pt"
                            )
        inputs = {k: v.cuda() for k, v in inputs.items()}
        output = self.encoder(**inputs, return_dict=True)
        query_emb = pooling(output.pooler_output, 
                            output.last_hidden_state, 
                            inputs['attention_mask'],
                            self.pooling_method)
        if  "dpr" in self.retrieval_method:
            query_emb = query.detach().cpu().numpy()
        else:
            query_emb = torch.nn.functional.normalize(query_emb, dim=-1).detach().cpu().numpy()
        query_emb = query_emb.astype(np.float32)
        return query_emb

    def load_docs(self, doc_idxs, content_function=base_content_function):
        results = [self.corpus.get(str(idx)) for idx in doc_idxs]
        # add content field
        for item in results:
            if 'contents' not in item:
                item['contents'] = content_function(item)
        return results

    def search(self, query: str, num: int = None) -> List[Dict[str, str]]:
        if num is None:
            num = self.topk
        query_emb = self._encode(query)
        scores, idxs = self.index.search(query_emb, k=num)
        idxs = idxs[0]
        results = self.load_docs(idxs, content_function=base_content_function)
        return results
