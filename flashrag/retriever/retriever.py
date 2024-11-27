import json
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
import warnings
from typing import List, Dict, Union
import functools
from tqdm import tqdm
import faiss
import copy
import numpy as np
from flashrag.utils import get_reranker
from flashrag.retriever.utils import load_corpus, load_docs, convert_numpy
from flashrag.retriever.encoder import Encoder, STEncoder, ClipEncoder


def cache_manager(func):
    """
    Decorator used for retrieving document cache.
    With the decorator, The retriever can store each retrieved document as a file and reuse it.
    """

    @functools.wraps(func)
    def wrapper(self, query_list, num=None, return_score=False):
        if num is None:
            num = self.topk
        if self.use_cache:
            if isinstance(query_list, str):
                new_query_list = [query_list]
            else:
                new_query_list = query_list

            no_cache_query = []
            cache_results = []
            for query in new_query_list:
                if query in self.cache:
                    cache_res = self.cache[query]
                    if len(cache_res) < num:
                        warnings.warn(f"The number of cached retrieval results is less than topk ({num})")
                    cache_res = cache_res[:num]
                    # separate the doc score
                    doc_scores = [item["score"] for item in cache_res]
                    cache_results.append((cache_res, doc_scores))
                else:
                    cache_results.append(None)
                    no_cache_query.append(query)

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
            results, scores = func(self, query_list, num, True)

        if self.save_cache:
            # merge result and score
            save_results = results.copy()
            save_scores = scores.copy()
            if isinstance(query_list, str):
                query_list = [query_list]
                if "batch" not in func.__name__:
                    save_results = [save_results]
                    save_scores = [save_scores]
            for query, doc_items, doc_scores in zip(query_list, save_results, save_scores):
                for item, score in zip(doc_items, doc_scores):
                    item["score"] = score
                self.cache[query] = doc_items

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
    def wrapper(self, query_list, num=None, return_score=False):
        results, scores = func(self, query_list, num, True)
        if self.use_reranker:
            results, scores = self.reranker.rerank(query_list, results)
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

    def _batch_search(self, query_list, num, return_score):
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
            self.searcher = bm25s.BM25.load(self.index_path, mmap=True, load_corpus=True)
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
            results = results[0]
            scores = scores[0]
        else:
            assert False, "Invalid bm25 backend!"

        if return_score:
            return results, scores
        else:
            return results

    def _batch_search(self, query_list, num: int = None, return_score=False, batch_size=None):
        if self.backend == "pyserini":
            # TODO: modify batch method
            results = []
            scores = []
            for query in query_list:
                item_result, item_score = self._search(query, num, True)
                results.append(item_result)
                scores.append(item_score)
        elif self.backend == "bm25s":
            import bm25s

            query_tokens = bm25s.tokenize(query_list, stemmer=self.stemmer)
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

    def _batch_search(self, query_list: List[str], num: int = None, return_score=False, batch_size=None):
        if isinstance(query_list, str):
            query_list = [query_list]
        if num is None:
            num = self.topk
        if batch_size is None:
            batch_size = self.batch_size

        results = []
        scores = []

        emb = self.encoder.encode(query_list, batch_size=batch_size, is_query=True)
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

    def _batch_search(
        self,
        query_list: List[str],
        target_modal: str = "text",
        num: int = None,
        return_score=False,
        batch_size=None
    ):
        if isinstance(query_list, str):
            query_list = [query_list]
        if num is None:
            num = self.topk
        if batch_size is None:
            batch_size = self.batch_size
        assert target_modal in ["image", "text"]

        query_modal = self._judge_input_modal(query_list[0])
        if query_modal == "image" and isinstance(query_list[0], str):
            from PIL import Image
            import requests

            query_list = [Image.open(requests.get(query, stream=True).raw) for query in query_list]


        results = []
        scores = []

        for start_idx in tqdm(range(0, len(query_list), batch_size), desc="Retrieval process: "):
            query_batch = query_list[start_idx : start_idx + batch_size]
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
        self.merge_method = config["multi_retriever_setting"]["merge_method"]  # concat

        self.retriever_list = self.load_all_retriever(config)
        self.config = config

    def load_all_retriever(self, config):
        retriever_config_list = config["multi_retriever_setting"]["retriever_list"]
        # use the same corpus for efficient memory usage
        all_corpus_dict = {}
        retriever_list = []
        for retriever_config in retriever_config_list:
            retrieval_method = retriever_config["retrieval_method"]
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

    def add_source(self, output_list: Union[list, tuple], retrieval_method):
        if isinstance(output_list, tuple):
            result, score = output_list[0], output_list[1]
            assert len(result) == len(score)
            for item in result:
                item['source'] = retrieval_method
            return result, score
        else:
            result = output_list
            for item in result:
                item['source'] = retrieval_method
            return result

    def _search_or_batch_search(self, query, target_modal, num, return_score, method, batch_size=None):
        if num is None:
            num_list = [retriever.topk for retriever in self.retriever_list]
        elif isinstance(num, int):
            num_list = [num] * len(self.retriever_list)
        else:
            assert len(num) == len(self.retriever_list)
            num_list = num
        if self.merge_method == "concat":
            output_list = []
            for num, retriever in zip(num_list, self.retriever_list):
                is_multimodal = isinstance(retriever, MultiModalRetriever)
                if method == 'search':
                    output = retriever.search(
                        query, 
                        target_modal=target_modal if is_multimodal else None, 
                        num=num, 
                        return_score=return_score
                    )
                else:
                    output = retriever.batch_search(
                        query, 
                        target_modal=target_modal if is_multimodal else None, 
                        num=num, 
                        return_score=return_score, 
                        batch_size=batch_size
                    )
                output = self.add_source(output, retriever.retrieval_method)
                output_list.append(output)
        else:
            raise NotImplementedError

        if return_score:
            result = sum([item[0] for item in output_list], [])
            score = sum([item[1] for item in output_list], [])
            return result, score
        else:
            return sum(output_list, [])

    def search(self, query, target_modal="text", num: Union[list, int, None] = None, return_score=False):
        return self._search_or_batch_search(query, target_modal, num, return_score, method='search')

    def batch_search(self, query, target_modal="text", num: Union[list, int, None] = None, return_score=False, batch_size=None):
        return self._search_or_batch_search(query, target_modal, num, return_score, method='search', batch_size=batch_size)
