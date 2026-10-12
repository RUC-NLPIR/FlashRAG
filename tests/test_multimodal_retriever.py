import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import faiss
import numpy as np

from datasets import Dataset
from flashrag.retriever import MultiModalRetriever


class TestMultiModalRetriever(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def retriever(self, index, vectors, corpus_size=None, batch_size=8):
        index_path = self.root / "text_index.faiss"
        faiss.write_index(index, str(index_path))
        if corpus_size is None:
            corpus_size = index.ntotal
        config = {
            "retrieval_method": "clip",
            "retrieval_topk": 4,
            "index_path": None,
            "corpus_path": None,
            "save_retrieval_cache": False,
            "use_retrieval_cache": False,
            "retrieval_cache_path": str(self.root / "retrieval_cache.json"),
            "save_dir": str(self.root),
            "use_reranker": False,
            "faiss_gpu": False,
            "retrieval_batch_size": batch_size,
            "retrieval_model_path": None,
            "multimodal_index_path_dict": {"text": str(index_path), "image": None},
        }
        corpus = Dataset.from_dict(
            {
                "id": [str(i) for i in range(corpus_size)],
                "contents": [f"Document {i}" for i in range(corpus_size)],
            }
        )
        with patch("flashrag.retriever.retriever.ClipEncoder"):
            retriever = MultiModalRetriever(config, corpus=corpus)
        retriever.encoder = Mock()
        retriever.encoder.encode.side_effect = lambda query, **kwargs: np.asarray(
            [vectors[q] for q in ([query] if isinstance(query, str) else query)], dtype="float32"
        )
        return retriever

    def assert_result(self, result, expected_ids, expected_scores, return_score, batch):
        docs, scores = result if return_score else (result, None)
        if not batch:
            docs = [docs]
            scores = [scores] if return_score else None
        self.assertEqual([[doc["id"] for doc in row] for row in docs], expected_ids)
        if return_score:
            self.assertEqual(scores, expected_scores)

    def test_flat_index_preserves_valid_results_and_scores(self):
        vectors = {"first": [1, 0], "second": [0, 1]}
        for index_class in [faiss.IndexFlatIP, faiss.IndexFlatL2]:
            index = index_class(2)
            index.add(np.asarray(list(vectors.values()), dtype="float32"))
            retriever = self.retriever(index, vectors)
            row_scores = [1.0, 0.0] if index_class is faiss.IndexFlatIP else [0.0, 2.0]
            for num in [1, 2, None]:
                for return_score in [False, True]:
                    for batch in [False, True]:
                        with self.subTest(
                            metric=index_class.__name__, num=num, return_score=return_score, batch=batch
                        ):
                            queries = ["first", "second"] if batch else "first"
                            search = retriever.batch_search if batch else retriever.search
                            result = search(queries, num=num, return_score=return_score)
                            ids = [["0", "1"], ["1", "0"]] if batch else [["0", "1"]]
                            self.assert_result(
                                result,
                                [row[:num] for row in ids],
                                [row_scores[:num] for _ in ids],
                                return_score,
                                batch,
                            )

    def test_flat_index_does_not_pad_missing_neighbors(self):
        vectors = {"first": [1, 0], "second": [0, 1]}
        index = faiss.IndexFlatIP(2)
        index.add(np.asarray(list(vectors.values()), dtype="float32"))
        retriever = self.retriever(index, vectors)
        for return_score in [False, True]:
            for batch in [False, True]:
                with self.subTest(return_score=return_score, batch=batch):
                    queries = ["first", "second"] if batch else "first"
                    search = retriever.batch_search if batch else retriever.search
                    result = search(queries, num=4, return_score=return_score)
                    expected_ids = [["0", "1"], ["1", "0"]] if batch else [["0", "1"]]
                    expected_scores = [[1.0, 0.0], [1.0, 0.0]] if batch else [[1.0, 0.0]]
                    self.assert_result(result, expected_ids, expected_scores, return_score, batch)

    def test_empty_index_returns_no_documents(self):
        retriever = self.retriever(faiss.IndexFlatIP(2), {"first": [1, 0]}, corpus_size=1)
        for return_score in [False, True]:
            for batch in [False, True]:
                with self.subTest(return_score=return_score, batch=batch):
                    search = retriever.batch_search if batch else retriever.search
                    result = search(["first", "first"] if batch else "first", return_score=return_score)
                    expected = [[], []] if batch else [[]]
                    self.assert_result(result, expected, expected, return_score, batch)

    def test_ivf_search_keeps_documents_and_scores_aligned_per_query(self):
        vectors = {"first": [0, 0], "second": [10, 0], "empty": [20, 0]}
        index = faiss.IndexIVFFlat(faiss.IndexFlatL2(2), 2, 3)
        index.train(np.repeat(np.asarray(list(vectors.values()), dtype="float32"), 40, axis=0))
        index.add(np.asarray([[0, 0], [10, 0], [10, 1]], dtype="float32"))
        retriever = self.retriever(index, vectors)
        expected_ids = [["0"], ["1", "2"], []]
        expected_scores = [[0.0], [0.0, 1.0], []]
        for return_score in [False, True]:
            with self.subTest(return_score=return_score):
                result = retriever.batch_search(list(vectors), num=2, return_score=return_score)
                self.assert_result(result, expected_ids, expected_scores, return_score, True)
            for query, ids, scores in zip(vectors, expected_ids, expected_scores):
                with self.subTest(query=query, return_score=return_score):
                    result = retriever.search(query, num=2, return_score=return_score)
                    self.assert_result(result, [ids], [scores], return_score, False)


if __name__ == "__main__":
    unittest.main()
