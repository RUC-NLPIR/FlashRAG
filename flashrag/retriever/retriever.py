import json
import os
import time
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import warnings
from typing import List, Dict, Union
import functools
from tqdm import tqdm
import faiss
import copy
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from flashrag.utils import get_reranker
from flashrag.retriever.utils import load_corpus, load_docs, convert_numpy
from flashrag.retriever.encoder import Encoder, STEncoder, ClipEncoder


def cache_manager(func):
    """
    Decorator used for retrieving document cache.
    With the decorator, The retriever can store each retrieved document as a file and reuse it.
    """

    @functools.wraps(func)
    def wrapper(self, query=None, num=None, return_score=False):
        if num is None:
            num = self.topk
        if self.use_cache:
            if isinstance(query, str):
                new_query_list = [query]
            else:
                new_query_list = query

            no_cache_query = []
            cache_results = []
            for new_query in new_query_list:
                if new_query in self.cache:
                    cache_res = self.cache[new_query]
                    if len(cache_res) < num:
                        warnings.warn(f"The number of cached retrieval results is less than topk ({num})")
                    cache_res = cache_res[:num]
                    # separate the doc score
                    doc_scores = [item["score"] for item in cache_res]
                    cache_results.append((cache_res, doc_scores))
                else:
                    cache_results.append(None)
                    no_cache_query.append(new_query)

            if no_cache_query != []:
                # use batch search without decorator
                no_cache_results, no_cache_scores = self._batch_search_with_rerank(no_cache_query, num, True)
                no_cache_idx = 0
                for idx, res in enumerate(cache_results):
                    if res is None:
                        assert new_query_list[idx] == no_cache_query[no_cache_idx]
                        cache_results = (
                            no_cache_results[no_cache_idx],
                            no_cache_scores[no_cache_scores],
                        )
                        no_cache_idx += 1

            results, scores = (
                [t[0] for t in cache_results],
                [t[1] for t in cache_results],
            )

        else:
            results, scores = func(self, query=query, num=num, return_score=True)

        if self.save_cache:
            # merge result and score
            save_results = results.copy()
            save_scores = scores.copy()
            if isinstance(query, str):
                query = [query]
                if "batch" not in func.__name__:
                    save_results = [save_results]
                    save_scores = [save_scores]
            for new_query, doc_items, doc_scores in zip(query, save_results, save_scores):
                for item, score in zip(doc_items, doc_scores):
                    item["score"] = score
                self.cache[new_query] = doc_items

        if return_score:
            return results, scores
        else:
            return results

    return wrapper


def rerank_manager(func):
    """
    Decorator used for reranking retrieved documents.
    """

    @functools.wraps(func)
    def wrapper(self, query, num=None, return_score=False):
        results, scores = func(self, query=query, num=num, return_score=True)
        if self.use_reranker:
            results, scores = self.reranker.rerank(query, results)
            if "batch" not in func.__name__:
                results = results[0]
                scores = scores[0]
        if return_score:
            return results, scores
        else:
            return results

    return wrapper


class BaseRetriever:
    """Base object for all retrievers."""

    def __init__(self, config):
        self.config = config
        self.retrieval_method = config["retrieval_method"]
        self.topk = config["retrieval_topk"]

        self.index_path = config["index_path"]
        self.corpus_path = config["corpus_path"]

        self.save_cache = config["save_retrieval_cache"]
        self.use_cache = config["use_retrieval_cache"]
        self.cache_path = config["retrieval_cache_path"]

        self.use_reranker = config["use_reranker"]
        if self.use_reranker:
            self.reranker = get_reranker(config)

        if self.save_cache:
            self.cache_save_path = os.path.join(config["save_dir"], "retrieval_cache.json")
            self.cache = {}
        if self.use_cache:
            assert self.cache_path is not None
            with open(self.cache_path, "r") as f:
                self.cache = json.load(f)

    def _save_cache(self):
        self.cache = convert_numpy(self.cache)

        def custom_serializer(obj):
            if isinstance(obj, np.float32):
                return float(obj)
            raise TypeError(f"Type {type(obj)} not serializable")

        with open(self.cache_save_path, "w") as f:
            json.dump(self.cache, f, indent=4, default=custom_serializer)

    def _search(self, query: str, num: int, return_score: bool) -> List[Dict[str, str]]:
        r"""Retrieve topk relevant documents in corpus.

        Return:
            list: contains information related to the document, including:
                contents: used for building index
                title: (if provided)
                text: (if provided)

        """

        pass

    def _batch_search(self, query, num, return_score):
        pass

    def search(self, *args, **kwargs):
        return self._search(*args, **kwargs)

    def batch_search(self, *args, **kwargs):
        return self._batch_search(*args, **kwargs)


class BaseTextRetriever(BaseRetriever):
    """Base text retriever."""

    def __init__(self, config):
        super().__init__(config)

    @cache_manager
    @rerank_manager
    def search(self, *args, **kwargs):
        return self._search(*args, **kwargs)

    @cache_manager
    @rerank_manager
    def batch_search(self, *args, **kwargs):
        return self._batch_search(*args, **kwargs)

    @rerank_manager
    def _batch_search_with_rerank(self, *args, **kwargs):
        return self._batch_search(*args, **kwargs)

    @rerank_manager
    def _search_with_rerank(self, *args, **kwargs):
        return self._search(*args, **kwargs)


class BM25Retriever(BaseTextRetriever):
    r"""BM25 retriever based on pre-built pyserini index."""

    def __init__(self, config, corpus=None):
        super().__init__(config)
        self.backend = config["bm25_backend"]

        if self.backend == "pyserini":
            # Warning: the method based on pyserini will be deprecated
            from pyserini.search.lucene import LuceneSearcher

            self.searcher = LuceneSearcher(self.index_path)
            self.contain_doc = self._check_contain_doc()
            if not self.contain_doc:
                if corpus is None:
                    self.corpus = load_corpus(self.corpus_path)
                else:
                    self.corpus = corpus
            self.max_process_num = 8
        elif self.backend == "bm25s":
            import Stemmer
            import bm25s

            self.stemmer = Stemmer.Stemmer("english")
            self.searcher = bm25s.BM25.load(self.index_path, mmap=True, load_corpus=False)
            self.corpus = load_corpus(self.corpus_path)
            self.searcher.corpus = self.corpus
            self.searcher.backend = "numba"

        else:
            assert False, "Invalid bm25 backend!"

    def _check_contain_doc(self):
        r"""Check if the index contains document content"""
        return self.searcher.doc(0).raw() is not None

    def _search(self, query: str, num: int = None, return_score=False) -> List[Dict[str, str]]:
        if num is None:
            num = self.topk
        if self.backend == "pyserini":
            hits = self.searcher.search(query, num)
            if len(hits) < 1:
                if return_score:
                    return [], []
                else:
                    return []

            scores = [hit.score for hit in hits]
            if len(hits) < num:
                warnings.warn("Not enough documents retrieved!")
            else:
                hits = hits[:num]

            if self.contain_doc:
                all_contents = [json.loads(self.searcher.doc(hit.docid).raw())["contents"] for hit in hits]
                results = [
                    {
                        "title": content.split("\n")[0].strip('"'),
                        "text": "\n".join(content.split("\n")[1:]),
                        "contents": content,
                    }
                    for content in all_contents
                ]
            else:
                results = load_docs(self.corpus, [hit.docid for hit in hits])
        elif self.backend == "bm25s":
            import bm25s

            query_tokens = bm25s.tokenize([query], stemmer=self.stemmer)
            results, scores = self.searcher.retrieve(query_tokens, k=num)
            results = list(results[0])
            scores = list(scores[0])
        else:
            assert False, "Invalid bm25 backend!"

        if return_score:
            return results, scores
        else:
            return results

    def _batch_search(self, query, num: int = None, return_score=False):
        if self.backend == "pyserini":
            # TODO: modify batch method
            results = []
            scores = []
            for _query in query:
                item_result, item_score = self._search(_query, num, True)
                results.append(item_result)
                scores.append(item_score)
        elif self.backend == "bm25s":
            import bm25s

            query_tokens = bm25s.tokenize(query, stemmer=self.stemmer)
            results, scores = self.searcher.retrieve(query_tokens, k=num)
        else:
            assert False, "Invalid bm25 backend!"

        if return_score:
            return results, scores
        else:
            return results


class DenseRetriever(BaseTextRetriever):
    r"""Dense retriever based on pre-built faiss index."""

    def __init__(self, config: dict, corpus=None):
        super().__init__(config)
        if self.index_path is None or not os.path.exists(self.index_path):
            raise Warning(f"Index file {self.index_path} does not exist!")
        self.index = faiss.read_index(self.index_path)
        if config["faiss_gpu"]:
            co = faiss.GpuMultipleClonerOptions()
            co.useFloat16 = True
            co.shard = True
            self.index = faiss.index_cpu_to_all_gpus(self.index, co=co)

        if corpus is None:
            self.corpus = load_corpus(self.corpus_path)
        else:
            self.corpus = corpus
        self.topk = config["retrieval_topk"]
        self.batch_size = config["retrieval_batch_size"]
        self.instruction = config["instruction"]

        if config["use_sentence_transformer"]:
            self.encoder = STEncoder(
                model_name=self.retrieval_method,
                model_path=config["retrieval_model_path"],
                max_length=config["retrieval_query_max_length"],
                use_fp16=config["retrieval_use_fp16"],
                instruction=self.instruction,
            )
        else:
            self.encoder = Encoder(
                model_name=self.retrieval_method,
                model_path=config["retrieval_model_path"],
                pooling_method=config["retrieval_pooling_method"],
                max_length=config["retrieval_query_max_length"],
                use_fp16=config["retrieval_use_fp16"],
                instruction=self.instruction,
            )

    def _search(self, query: str, num: int = None, return_score=False):
        if num is None:
            num = self.topk
        query_emb = self.encoder.encode(query)
        scores, idxs = self.index.search(query_emb, k=num)
        scores = scores.tolist()
        idxs = idxs[0]
        scores = scores[0]

        results = load_docs(self.corpus, idxs)
        if return_score:
            return results, scores
        else:
            return results

    def _batch_search(self, query: List[str], num: int = None, return_score=False):
        if isinstance(query, str):
            query = [query]
        if num is None:
            num = self.topk
        batch_size = self.batch_size

        results = []
        scores = []

        emb = self.encoder.encode(query, batch_size=batch_size, is_query=True)
        print("Begin faiss searching...")
        scores, idxs = self.index.search(emb, k=num)
        print("End faiss searching")
        scores = scores.tolist()
        idxs = idxs.tolist()

        flat_idxs = sum(idxs, [])
        results = load_docs(self.corpus, flat_idxs)
        results = [results[i * num : (i + 1) * num] for i in range(len(idxs))]

        if return_score:
            return results, scores
        else:
            return results


class MultiModalRetriever(BaseRetriever):
    r"""Multi-modal retriever based on pre-built faiss index."""

    def __init__(self, config: dict, corpus=None):
        super().__init__(config)
        self.mm_index_dict = config[
            "multimodal_index_path_dict"
        ]  # {"text": "path/to/text_index", "image": "path/to/image_index"}
        self.index_dict = {"text": None, "image": None}
        for modal in ["text", "image"]:
            idx_path = self.mm_index_dict[modal]
            if idx_path is not None:
                self.index_dict[modal] = faiss.read_index(idx_path)
            if config["faiss_gpu"]:
                co = faiss.GpuMultipleClonerOptions()
                co.useFloat16 = True
                co.shard = True
                self.index_dict[modal] = faiss.index_cpu_to_all_gpus(self.index_dict[modal], co=co)
        if corpus is None:
            self.corpus = load_corpus(self.corpus_path)
        else:
            self.corpus = corpus
        self.topk = config["retrieval_topk"]
        self.batch_size = config["retrieval_batch_size"]

        self.encoder = ClipEncoder(
            model_name=self.retrieval_method,
            model_path=config["retrieval_model_path"],
        )

    def _judge_input_modal(self, query):
        if not isinstance(query, str):
            return "image"
        else:
            if query.startswith("http") or query.endswith(".jpg") or query.endswith(".png"):
                return "image"
            else:
                return "text"

    def _search(self, query, target_modal: str = "text", num: int = None, return_score=False):
        if num is None:
            num = self.topk
        assert target_modal in ["image", "text"]

        query_modal = (
            self._judge_input_modal(query) if not isinstance(query, list) else self._judge_input_modal(query[0])
        )
        if query_modal == "image" and isinstance(query, str):
            from PIL import Image
            import requests

            query = Image.open(requests.get(query, stream=True).raw)

        query_emb = self.encoder.encode(query, modal=query_modal)

        scores, idxs = self.index_dict[target_modal].search(query_emb, k=num)
        scores = scores.tolist()
        idxs = idxs[0]
        scores = scores[0]

        results = load_docs(self.corpus, idxs)
        if return_score:
            return results, scores
        else:
            return results

    def _batch_search(self, query: List[str], target_modal: str = "text", num: int = None, return_score=False):
        if isinstance(query, str):
            query = [query]
        if num is None:
            num = self.topk
        batch_size = self.batch_size
        assert target_modal in ["image", "text"]

        query_modal = self._judge_input_modal(query[0])
        if query_modal == "image" and isinstance(query[0], str):
            from PIL import Image
            import requests

            query = [Image.open(requests.get(query, stream=True).raw) for query in query]

        results = []
        scores = []

        for start_idx in tqdm(range(0, len(query), batch_size), desc="Retrieval process: "):
            query_batch = query[start_idx : start_idx + batch_size]
            batch_emb = self.encoder.encode(query_batch, modal=query_modal)
            batch_scores, batch_idxs = self.index_dict[target_modal].search(batch_emb, k=num)
            batch_scores = batch_scores.tolist()
            batch_idxs = batch_idxs.tolist()

            flat_idxs = sum(batch_idxs, [])
            batch_results = load_docs(self.corpus, flat_idxs)
            batch_results = [batch_results[i * num : (i + 1) * num] for i in range(len(batch_idxs))]

            scores.extend(batch_scores)
            results.extend(batch_results)

        if return_score:
            return results, scores
        else:
            return results


class MultiRetrieverRouter:
    def __init__(self, config):
        self.merge_method = config["multi_retriever_setting"].get("merge_method", "concat")  # concat/rrf/rerank
        self.final_topk = config["multi_retriever_setting"].get("topk", 5)
        self.retriever_list = self.load_all_retriever(config)
        self.config = config

        if self.merge_method == 'rerank':
            config['multi_retriever_setting']['rerank_topk'] = self.final_topk
            config['multi_retriever_setting']['device'] = config['device']
            self.reranker = get_reranker(config['multi_retriever_setting'])

    def load_all_retriever(self, config):
        retriever_config_list = config["multi_retriever_setting"]["retriever_list"]
        # use the same corpus for efficient memory usage
        all_corpus_dict = {}
        retriever_list = []
        for retriever_config in retriever_config_list:
            retrieval_method = retriever_config["retrieval_method"]
            print(f"Loading {retrieval_method} retriever...")
            retrieval_model_path = retriever_config["retrieval_model_path"]
            corpus_path = retriever_config["corpus_path"]

            if retrieval_method == "bm25":
                if corpus_path is None:
                    corpus = None
                else:
                    if corpus_path in all_corpus_dict:
                        corpus = all_corpus_dict[corpus_path]
                    else:
                        corpus = load_corpus(corpus_path)
                        all_corpus_dict[corpus_path] = corpus
                retriever = BM25Retriever(retriever_config, corpus)
            else:
                if corpus_path in all_corpus_dict:
                    corpus = all_corpus_dict[corpus_path]
                else:
                    corpus = load_corpus(corpus_path)
                    all_corpus_dict[corpus_path] = corpus

                # judge modality
                from transformers import AutoConfig

                try:
                    model_config = AutoConfig.from_pretrained(retrieval_model_path)
                    arch = model_config.architectures[0]
                    if "clip" in arch.lower():
                        retriever = MultiModalRetriever(retriever_config, corpus)
                    else:
                        retriever = DenseRetriever(retriever_config, corpus)
                except:
                    retriever = DenseRetriever(retriever_config, corpus)

            retriever_list.append(retriever)

        return retriever_list

    def add_source(self, result: Union[list, tuple], retriever):
        retrieval_method = retriever.retrieval_method
        corpus_path = retriever.corpus_path
        is_multimodal = isinstance(retriever, MultiModalRetriever)
        # for naive search, result is a list of dict, each repr a doc
        # for batch search, result is a list of list, each repr a doc list(per query)
        for item in result:
            if isinstance(item, list):
                for _item in item:
                    _item["source"] = retrieval_method
                    _item["corpus_path"] = corpus_path
                    _item['is_multimodal'] = is_multimodal
            else:
                item["source"] = retrieval_method
                item["corpus_path"] = corpus_path
                item['is_multimodal'] = is_multimodal
        return result

    def _search_or_batch_search(self, query: Union[str, list], target_modal, num, return_score, method):
        if num is None:
            num = self.final_topk

        result_list = []
        score_list = []

        def process_retriever(retriever):
            is_multimodal = isinstance(retriever, MultiModalRetriever)
            params = {"query": query, "return_score": return_score}
            if is_multimodal:
                params["target_modal"] = target_modal

            if method == "search":
                output = retriever.search(**params)
            else:
                output = retriever.batch_search(**params)

            if return_score:
                result, score = output
            else:
                result = output
                score = None

            result = self.add_source(result, retriever)
            return result, score

        with ThreadPoolExecutor(max_workers=4) as executor:
            future_to_retriever = {executor.submit(process_retriever, retriever): retriever for retriever in self.retriever_list}
            for future in as_completed(future_to_retriever):
                try:
                    result, score = future.result()
                    result_list.extend(result)
                    if score is not None:
                        score_list.extend(score)
                except Exception as e:
                    print(f"Error processing retriever {future_to_retriever[future]}: {e}")

        result_list, score_list = self.reorder(result_list, score_list)
        result_list, score_list = self.post_process_result(query, result_list, score_list, num)
        if return_score:
            return result_list, score_list
        else:
            return result_list


    def reorder(self, result_list, score_list):
        """
        batch_search:
        original result like: [[bm25-q1-d1, bm25-q1-d2],[bm25-q2-d1, bm25-q2-d2], [e5-q1-d1, e5-q1-d2], [e5-q2-d1, e5-q2-d2]]
        reorder to: [[bm25-q1-d1, bm25-q1-d2, e5-q1-d1, e5-q1-d2], [bm25-q2-d1,bm25-q2-d2, e5-q2-d1, e5-q2-d2]]

        navie search:
        original result like: [bm25-d1, bm25-d2, e5-d1, e5-d2]
        
        """

        retriever_num = len(self.retriever_list)
        query_num = len(result_list) // retriever_num
        assert query_num * retriever_num == len(result_list)

        if isinstance(result_list[0], dict):
            return result_list, score_list

        final_result = []
        final_score = []
        for q_idx in range(query_num):
            final_result.append(sum([result_list[q_idx + r_idx * query_num] for r_idx in range(retriever_num)], []))
            if score_list != []:
                final_score.append(sum([score_list[q_idx + r_idx * query_num] for r_idx in range(retriever_num)], []))
        return final_result, final_score

    def post_process_result(self, query: Union[str, list], result_list, score_list, num):
        # based on self.merge_method
        if self.merge_method == "concat":
            # remove duplicate doc
            if isinstance(result_list[0], dict):
                exist_id = set()
                for idx, doc in enumerate(result_list):
                    if doc["id"] not in exist_id:
                        exist_id.add(doc["id"])
                    else:
                        result_list.remove(doc)
                        if score_list != []:
                            score_list.remove(idx)
            else:
                for query_idx, query_doc_list in enumerate(result_list):
                    exist_id = set()
                    for doc_idx, doc in enumerate(query_doc_list):
                        if doc not in exist_id:
                            exist_id.add(doc)
                        else:
                            query_doc_list.remove(doc)
                            if score_list != []:
                                score_list[query_idx].remove(doc_idx)
            return result_list, score_list
        elif self.merge_method == "rrf":
            if (isinstance(result_list[0], dict) and len(set([doc["corpus_path"] for doc in result_list])) > 1) or (
                isinstance(result_list[0], list) and len(set([doc["corpus_path"] for doc in result_list[0]])) > 1
            ):
                warnings.warn(
                    "Using multiple corpus may lead to conflicts in DOC IDs, which may result in incorrect rrf results!"
                )
            if isinstance(result_list[0], dict):
                result_list, score_list = self.rrf_merge([result_list], num, k=60)
                result_list = result_list[0]
                score_list = score_list[0]
            else:
                result_list, score_list = self.rrf_merge(result_list, num, k=60)
            return result_list, score_list
        elif self.merge_method == 'rerank':
            if isinstance(result_list[0], dict):
                query, result_list, score_list = [query], [result_list], [score_list]
            # parse the result of multimodal corpus
            for item_result in result_list:
                for item in item_result:
                    if item['is_multimodal']:
                        item['contents'] = item['text']
            # rerank all docs
            result_list, score_list = self.reranker.rerank(query, result_list, topk=num)
            if isinstance(query, str):
                result_list, score_list = result_list[0], score_list[0]
            return result_list, score_list
        else:
            raise NotImplementedError

    def rrf_merge(self, results, topk=10, k=60):
        """
        Perform Reciprocal Rank Fusion (RRF) on retrieval results.

        Args:
            results (list of list of dict): Retrieval results for multiple queries.
            topk (int): Number of top results to return per query.
            k (int): RRF hyperparameter to adjust rank contribution.

        Returns:
            list of list of dict: Fused results with topk highest scores per query.
        """
        fused_results = []
        fused_scores = []
        for query_results in results:
            # Initialize a score dictionary to accumulate RRF scores
            score_dict = {}
            retriever_result_dict = {}
            id2item = {}
            for item in query_results:
                source = item["source"]
                if source not in retriever_result_dict:
                    retriever_result_dict[source] = []
                retriever_result_dict[source].append(item["id"])
                id2item[item["id"]] = item

            # Calculate RRF scores for each document
            for retriever, retriever_result in retriever_result_dict.items():
                for rank, doc_id in enumerate(retriever_result, start=1):
                    if doc_id not in score_dict:
                        score_dict[doc_id] = 0
                    # Add RRF score for the document
                    score_dict[doc_id] += 1 / (k + rank)

            # Sort by accumulated RRF score
            sorted_results = sorted(score_dict.items(), key=lambda x: x[1], reverse=True)

            # Keep only the topk results
            top_ids = [i[0] for i in sorted_results[:topk]]
            top_scores = [i[1] for i in sorted_results[:topk]]

            fused_results.append([id2item[id] for id in top_ids])
            fused_scores.append(top_scores)

        return fused_results, fused_scores

    def search(self, query, target_modal="text", num: Union[list, int, None] = None, return_score=False):
        return self._search_or_batch_search(query, target_modal, num, return_score, method="search")

    def batch_search(self, query, target_modal="text", num: Union[list, int, None] = None, return_score=False):
        return self._search_or_batch_search(query, target_modal, num, return_score, method="batch_search")
