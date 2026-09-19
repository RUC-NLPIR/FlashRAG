import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from flashrag.retriever import BaseTextRetriever


class TestRetrieverCache(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.cache_path = Path(self.directory.name) / "retrieval_cache.json"
        self.docs = [
            {"id": "1", "contents": "First document", "score": 0.8},
            {"id": "2", "contents": "Second document", "score": 0.5},
        ]
        self.cache_path.write_text(json.dumps({"cached": self.docs, "empty": []}))

    def retriever(self, save_cache=False):
        return BaseTextRetriever(
            {
                "retrieval_method": "bm25",
                "retrieval_topk": 2,
                "index_path": None,
                "corpus_path": None,
                "save_retrieval_cache": save_cache,
                "use_retrieval_cache": True,
                "retrieval_cache_path": str(self.cache_path),
                "use_reranker": False,
                "save_dir": self.directory.name,
            }
        )

    def test_single_search_returns_flat_results_and_scores(self):
        for query in ["cached", "missing"]:
            for num in [1, 2]:
                for return_score in [False, True]:
                    with self.subTest(query=query, num=num, return_score=return_score):
                        retriever = self.retriever()
                        docs = copy.deepcopy(self.docs[:num])
                        scores = [doc["score"] for doc in docs]
                        with patch.object(retriever, "_batch_search", return_value=([docs], [scores])) as search:
                            result = retriever.search(query, num=num, return_score=return_score)
                        self.assertEqual(result, (docs, scores) if return_score else docs)
                        if query == "cached":
                            search.assert_not_called()
                        else:
                            search.assert_called_once_with(query=[query], num=num, return_score=True)

    def test_single_search_can_save_cache(self):
        for query in ["cached", "missing"]:
            with self.subTest(query=query):
                retriever = self.retriever(save_cache=True)
                docs = copy.deepcopy(self.docs)
                scores = [doc["score"] for doc in docs]
                with patch.object(retriever, "_batch_search", return_value=([docs], [scores])):
                    self.assertEqual(retriever.search(query), docs)
                retriever._save_cache()
                saved = json.loads(self.cache_path.read_text())
                self.assertEqual(saved[query], docs)

    def test_single_search_with_no_cached_results(self):
        for return_score in [False, True]:
            with self.subTest(return_score=return_score):
                with self.assertWarns(UserWarning):
                    result = self.retriever().search("empty", return_score=return_score)
                self.assertEqual(result, ([], []) if return_score else [])

    def test_batch_search_keeps_query_dimension_and_order(self):
        for return_score in [False, True]:
            with self.subTest(return_score=return_score):
                retriever = self.retriever()
                missing_docs = [{"id": "3", "contents": "Another document"}]
                with patch.object(retriever, "_batch_search", return_value=([missing_docs], [[0.7]])) as search:
                    result = retriever.batch_search(["missing", "cached"], return_score=return_score)
                expected_docs = [missing_docs, self.docs]
                expected_scores = [[0.7], [0.8, 0.5]]
                self.assertEqual(result, (expected_docs, expected_scores) if return_score else expected_docs)
                search.assert_called_once_with(query=["missing"], num=2, return_score=True)


if __name__ == "__main__":
    unittest.main()
