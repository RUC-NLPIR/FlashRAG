import faiss
import json
from abc import ABC, abstractmethod
from typing import List, Dict
import numpy as np
import torch
from tqdm import tqdm
from multiprocessing import Pool
from pyserini.search.lucene import LuceneSearcher
from pyserini.index.lucene import IndexReader

from torch import Tensor
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModel
from flashrag.retriever.utils import load_model, pooling, base_content_function, load_database, load_docs

        
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
        self.max_process_num = 8
        
    def _check_contain_doc(self):
        r"""Check if the index contains document content
        """
        return self.searcher.doc(0).raw() is not None

    def search(self, query: str, num: int = None, return_score = False) -> List[Dict[str, str]]:
        if num is None:
            num = self.topk
        hits = self.searcher.search(query, num)
        if len(hits) < 1:
            if self.return_score:
                return [],[]
            else:
                return []
            
        scores = [hit.score for hit in hits]

        if self.contain_doc:
            all_contents = [json.loads(self.searcher.doc(hits[i].docid).raw())['contents'] for i in range(num)]
            results = [{'title': content.split("\n")[0].strip("\""), 
                        'text': "\n".join(content.split("\n")[1:]),
                        'contents': content} for content in all_contents]
        else:
            results = load_docs(self.corpus, [hits[i].docid for i in range(num)])

        if return_score:
            return results, scores
        else:
            return results

    def batch_search(self, query_list, num: int = None, batch_size = None, return_score = False):
        # TODO: modify batch method
        results = []
        scores = []
        for query in query_list:
            item_result, item_score = self.search(query, num=num,return_score=True)
            results.append(item_result)
            scores.append(item_score)

        if return_score:
            return results, scores
        else:
            return results

class DenseRetriever(BaseRetriever):
    r"""Dense retriever based on pre-built faiss index."""

    def __init__(self, config: dict):
        super().__init__(config)
        self.index = faiss.read_index(self.index_path)
        self.index = faiss.index_cpu_to_all_gpus(self.index)
        self.corpus = load_database(self.corpus_database_path)
        self.encoder, self.tokenizer = load_model(model_path = config['retrieval_model_path'], 
                                                  use_fp16 = config['retrieval_use_fp16'])
        self.topk = config['retrieval_topk']
        self.pooling_method = self.config['retrieval_pooling_method'] 
        self.query_max_length = self.config['retrieval_query_max_length']
        self.batch_size = self.config['retrieval_batch_size']

    
    @torch.no_grad()
    def _encode(self, query_list, is_query=True) -> np.ndarray:
        # processing query for different encoders
        if isinstance(query_list, str):
            query_list = [query_list]

        if "e5" in self.retrieval_method.lower():
            if is_query:
                query_list = [f"query: {query}" for query in query_list]
            else:
                query_list = [f"passage: {query}" for query in query_list]

        inputs = self.tokenizer(query_list, 
                                max_length = self.query_max_length, 
                                padding = True, 
                                truncation = True, 
                                return_tensors = "pt"
                            )
        inputs = {k: v.cuda() for k, v in inputs.items()}

        #TODO: support encoder-only T5 model
        if "T5" in type(self.encoder).__name__:
            # T5-based retrieval model
            decoder_input_ids = torch.zeros(
                (inputs['input_ids'].shape[0], 1), dtype=torch.long
            ).to(inputs['input_ids'].device)
            output = self.encoder(
                **inputs, decoder_input_ids=decoder_input_ids, return_dict=True
            )
            query_emb = output.last_hidden_state[:, 0, :]

        else:
            output = self.encoder(**inputs, return_dict=True)
            query_emb = pooling(output.pooler_output, 
                                output.last_hidden_state, 
                                inputs['attention_mask'],
                                self.pooling_method)
            if  "dpr" not in self.retrieval_method:
                query_emb = torch.nn.functional.normalize(query_emb, dim=-1)

        query_emb = query_emb.detach().cpu().numpy()
        query_emb = query_emb.astype(np.float32)
        return query_emb

    
    def search(self, query, num: int = None, return_score = False):
        if num is None:
            num = self.topk
        query_emb = self._encode(query)
        scores, idxs = self.index.search(query_emb, k=num)
        idxs = idxs[0]
        scores = scores[0]

        results = load_docs(self.corpus, idxs, content_function=base_content_function)
        if return_score:
            return results, scores
        else:
            return results


    def batch_search(self, query_list, num: int = None, batch_size = None, return_score = False):
        if num is None:
            num = self.topk
        if batch_size is None:
            batch_size = self.batch_size

        results = []
        scores = []

        for start_idx in tqdm(range(0, len(query_list), batch_size), desc='Retrieval process: '):
            query_batch = query_list[start_idx:start_idx + batch_size]
            batch_emb = self._encode(query_batch)
            batch_scores, batch_idxs = self.index.search(batch_emb, k=num)

            batch_scores = batch_scores.tolist()
            batch_idxs = batch_idxs.tolist()
            
            flat_idxs = sum(batch_idxs, [])
            batch_results = load_docs(self.corpus, flat_idxs, content_function=base_content_function)
            
            batch_results = [batch_results[i*num : (i+1)*num] for i in range(len(batch_idxs))]
            
            scores.extend(batch_scores)
            results.extend(batch_results)
        
        if return_score:
            return results, scores
        else:
            return results
        
        
