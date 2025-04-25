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
from flashrag.retriever.utils import load_corpus, load_docs, convert_numpy, judge_image, judge_zh
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
                        cache_results[idx] = (
                            no_cache_results[no_cache_idx],
                            no_cache_scores[no_cache_idx],
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
        self._config = config
        self.update_config()

    @property
    def config(self):
        return self._config

    @config.setter
    def config(self, config_data):
        self._config = config_data
        self.update_config()

    def update_config(self):
        self.update_base_setting()
        self.update_additional_setting()

    def update_base_setting(self):
        self.retrieval_method = self._config["retrieval_method"]
        self.topk = self._config["retrieval_topk"]

        self.index_path = self._config["index_path"]
        self.corpus_path = self._config["corpus_path"]

        self.save_cache = self._config["save_retrieval_cache"]
        self.use_cache = self._config["use_retrieval_cache"]
        self.cache_path = self._config["retrieval_cache_path"]

        self.use_reranker = self._config["use_reranker"]
        if self.use_reranker:
            self.reranker = get_reranker(self._config)
        else:
            self.reranker = None

        if self.save_cache:
            self.cache_save_path = os.path.join(self._config["save_dir"], "retrieval_cache.json")
            self.cache = {}
        if self.use_cache:
            assert self.cache_path is not None
            with open(self.cache_path, "r") as f:
                self.cache = json.load(f)
        self.silent = self._config["silent_retrieval"] if "silent_retrieval" in self._config else False

    def update_additional_setting(self):
        pass

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
        self.load_model_corpus(corpus)

    def update_additional_setting(self):
        self.backend = self._config["bm25_backend"]

    def load_model_corpus(self, corpus):
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

            self.corpus = load_corpus(self.corpus_path)
            is_zh = judge_zh(self.corpus[0]["contents"])

            self.searcher = bm25s.BM25.load(self.index_path, mmap=True, load_corpus=False)
            if is_zh:
                self.tokenizer = bm25s.tokenization.Tokenizer(stopwords="zh")
                self.tokenizer.load_stopwords(self.index_path)
                self.tokenizer.load_vocab(self.index_path)
            else:
                stemmer = Stemmer.Stemmer("english")
                self.tokenizer = bm25s.tokenization.Tokenizer(stopwords="en", stemmer=stemmer)
                self.tokenizer.load_stopwords(self.index_path)
                self.tokenizer.load_vocab(self.index_path)

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
            is_zh = judge_zh(query)
            if is_zh:
                self.searcher.set_language("zh")
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
                        "id": hit.docid, 
                        "title": content.split("\n")[0].strip('"'),
                        "text": "\n".join(content.split("\n")[1:]),
                        "contents": content,
                    }
                    for content, hit in zip(all_contents, hits)
                ]
            else:
                results = load_docs(self.corpus, [hit.docid for hit in hits])
        elif self.backend == "bm25s":
            # query_tokens = self.tokenizer.tokenize([query], return_as="tuple", update_vocab=False)
            query_tokens = bm25s.tokenize([query])
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
            # query_tokens = self.tokenizer.tokenize(query, return_as="tuple", update_vocab=False)
            query_tokens = bm25s.tokenize(query)
            results, scores = self.searcher.retrieve(query_tokens, k=num)
        else:
            assert False, "Invalid bm25 backend!"
        results = results.tolist() if isinstance(results, np.ndarray) else results
        scores = scores.tolist() if isinstance(scores, np.ndarray) else scores
        if return_score:
            return results, scores
        else:
            return results


class DenseRetriever(BaseTextRetriever):
    r"""Dense retriever based on pre-built faiss index."""

    def __init__(self, config: dict, corpus=None):
        super().__init__(config)

        self.load_corpus(corpus)
        self.load_index()
        self.load_model()

    def load_corpus(self, corpus):
        if corpus is None:
            self.corpus = load_corpus(self.corpus_path)
        else:
            self.corpus = corpus

    def load_index(self):
        if self.index_path is None or not os.path.exists(self.index_path):
            raise Warning(f"Index file {self.index_path} does not exist!")
        self.index = faiss.read_index(self.index_path)
        if self.use_faiss_gpu:
            co = faiss.GpuMultipleClonerOptions()
            co.useFloat16 = True
            co.shard = True
            self.index = faiss.index_cpu_to_all_gpus(self.index, co=co)

    def update_additional_setting(self):
        self.query_max_length = self._config["retrieval_query_max_length"]
        self.pooling_method = self._config["retrieval_pooling_method"]
        self.use_fp16 = self._config["retrieval_use_fp16"]
        self.batch_size = self._config["retrieval_batch_size"]
        self.instruction = self._config["instruction"]

        self.retrieval_model_path = self._config["retrieval_model_path"]
        self.use_st = self._config["use_sentence_transformer"]
        self.use_faiss_gpu = self._config["faiss_gpu"]

    def load_model(self):
        if self.use_st:
            self.encoder = STEncoder(
                model_name=self.retrieval_method,
                model_path=self._config["retrieval_model_path"],
                max_length=self.query_max_length,
                use_fp16=self.use_fp16,
                instruction=self.instruction,
                silent=self.silent,
            )
        else:
            # check pooling method
            self._check_pooling_method(self.retrieval_model_path, self.pooling_method)
            self.encoder = Encoder(
                model_name=self.retrieval_method,
                model_path=self.retrieval_model_path,
                pooling_method=self.pooling_method,
                max_length=self.query_max_length,
                use_fp16=self.use_fp16,
                instruction=self.instruction,
            )

    def _check_pooling_method(self, model_path, pooling_method):
        try:
            # read pooling method from 1_Pooling/config.json
            pooling_config = json.load(open(os.path.join(model_path, "1_Pooling/config.json")))
            for k, v in pooling_config.items():
                if k.startswith("pooling_mode") and v == True:
                    detect_pooling_method = k.split("pooling_mode_")[-1]
                    if detect_pooling_method == "mean_tokens":
                        detect_pooling_method = "mean"
                    elif detect_pooling_method == "cls_token":
                        detect_pooling_method = "cls"
                    else:
                        # raise warning: not implemented pooling method
                        warnings.warn(f"Pooling method {detect_pooling_method} is not implemented.", UserWarning)
                        detect_pooling_method = "mean"
                    break
        except:
            detect_pooling_method = None

        if detect_pooling_method is not None and detect_pooling_method != pooling_method:
            warnings.warn(
                f"Pooling method in model config file is {detect_pooling_method}, but the input is {pooling_method}. Please check carefully."
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
        scores, idxs = self.index.search(emb, k=num)
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
            model_name=self.retrieval_method, model_path=config["retrieval_model_path"], silent=self.silent
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

            if os.path.exists(query):
                query = Image.open(query)
            else:
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

            if os.path.exists(query[0]):
                query = [Image.open(q) for q in query]
            else:
                query = [Image.open(requests.get(q, stream=True).raw) for q in query]

        results = []
        scores = []

        for start_idx in tqdm(range(0, len(query), batch_size), desc="Retrieval process: ", disable=self.silent):
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

        if self.merge_method == "rerank":
            config["multi_retriever_setting"]["rerank_topk"] = self.final_topk
            config["multi_retriever_setting"]["device"] = config["device"]
            self.reranker = get_reranker(config["multi_retriever_setting"])

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
                    print("arch: ", arch)
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
                    _item["is_multimodal"] = is_multimodal
            else:
                item["source"] = retrieval_method
                item["corpus_path"] = corpus_path
                item["is_multimodal"] = is_multimodal
        return result

    def _search_or_batch_search(self, query: Union[str, list], target_modal, num, return_score, method, retriever_list):
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
            future_to_retriever = {
                executor.submit(process_retriever, retriever): retriever for retriever in retriever_list
            }
            for future in as_completed(future_to_retriever):
                try:
                    result, score = future.result()
                    result_list.extend(result)
                    if score is not None:
                        score_list.extend(score)
                except Exception as e:
                    print(f"Error processing retriever {future_to_retriever[future]}: {e}")
        result_list, score_list = self.reorder(result_list, score_list, retriever_list)
        result_list, score_list = self.post_process_result(query, result_list, score_list, num)
        if return_score:
            return result_list, score_list
        else:
            return result_list

    def reorder(self, result_list, score_list, retriever_list):
        """
        batch_search:
        original result like: [[bm25-q1-d1, bm25-q1-d2],[bm25-q2-d1, bm25-q2-d2], [e5-q1-d1, e5-q1-d2], [e5-q2-d1, e5-q2-d2]]
        reorder to: [[bm25-q1-d1, bm25-q1-d2, e5-q1-d1, e5-q1-d2], [bm25-q2-d1,bm25-q2-d2, e5-q2-d1, e5-q2-d2]]

        navie search:
        original result like: [bm25-d1, bm25-d2, e5-d1, e5-d2]

        """

        retriever_num = len(retriever_list)
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
                        if doc["id"] not in exist_id:
                            exist_id.add(doc["id"])
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
        elif self.merge_method == "rerank":
            if isinstance(result_list[0], dict):
                query, result_list, score_list = [query], [result_list], [score_list]
            # parse the result of multimodal corpus
            for item_result in result_list:
                for item in item_result:
                    if item["is_multimodal"]:
                        item["contents"] = item["text"]
            # rerank all docs
            print(result_list)
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
        # query: str or PIL.Image
        # judge query type: text or image
        if judge_image(query):
            retriever_list = [
                retriever for retriever in self.retriever_list if isinstance(retriever, MultiModalRetriever)
            ]
        else:
            retriever_list = self.retriever_list
        if target_modal == "image":
            # remove text retriever
            retriever_list = [retriever for retriever in retriever_list if isinstance(retriever, MultiModalRetriever)]

        return self._search_or_batch_search(
            query, target_modal, num, return_score, method="search", retriever_list=retriever_list
        )

    def batch_search(self, query, target_modal="text", num: Union[list, int, None] = None, return_score=False):
        # judge query type: text or image
        if not isinstance(query, list):
            query = [query]
        if target_modal == "image":
            self._retriever_list = [
                retriever for retriever in self.retriever_list if isinstance(retriever, MultiModalRetriever)
            ]
        else:
            self._retriever_list = self.retriever_list
        query_type_list = [judge_image(q) for q in query]
        if all(query_type_list):
            # all query is image
            if self.merge_method == "rerank":
                warnings.warn("merge_method is rerank, but all query is image, use default method `concat` instead")
                self.merge_method = "concat"
            retriever_list = [
                retriever for retriever in self._retriever_list if isinstance(retriever, MultiModalRetriever)
            ]

            return self._search_or_batch_search(
                query, target_modal, num, return_score, method="batch_search", retriever_list=retriever_list
            )
        elif all([not t for t in query_type_list]):
            # all query is text
            # if exist text retriever, don't use mm retriever for text-text search
            if any([isinstance(retriever, BaseTextRetriever) for retriever in self._retriever_list]):
                self._retriever_list = [
                    retriever for retriever in self._retriever_list if not isinstance(retriever, MultiModalRetriever)
                ]
            return self._search_or_batch_search(
                query, target_modal, num, return_score, method="batch_search", retriever_list=self._retriever_list
            )
        else:
            # query list is the mix of image and text
            if self.merge_method == "rerank":
                warnings.warn("merge_method is rerank, but some query is image, use default method `concat` instead")
                self.merge_method = "concat"
            image_query_idx = [i for i, t in enumerate(query_type_list) if t]
            image_query_list = [query[i] for i in image_query_idx]
            text_query_list = [q for q in query if q not in image_query_list]

            text_output = self._search_or_batch_search(
                text_query_list,
                target_modal,
                num,
                return_score,
                method="batch_search",
                retriever_list=self._retriever_list,
            )
            retriever_list = [
                retriever for retriever in self._retriever_list if isinstance(retriever, MultiModalRetriever)
            ]
            image_output = self._search_or_batch_search(
                text_query_list, target_modal, num, return_score, method="batch_search", retriever_list=retriever_list
            )

            # merge text output and image output
            if return_score:
                text_result, text_score = text_output
                image_result, image_score = image_output
                final_result = []
                final_score = []
                text_idx = 0
                image_idx = 0
                for idx in range(len(query)):
                    if idx not in image_query_idx:
                        final_result.append(text_result[text_idx])
                        final_score.append(text_score[text_idx])
                        text_idx += 1
                    else:
                        final_result.append(image_result[image_idx])
                        final_score.append(image_score[image_idx])
                        image_idx += 1
                return final_result, final_score
            else:
                final_result = []
                text_idx = 0
                image_idx = 0
                for idx in range(len(query)):
                    if idx not in image_query_idx:
                        final_result.append(text_result[text_idx])
                        text_idx += 1
                    else:
                        final_result.append(image_result[image_idx])
                        image_idx += 1
                return final_result
