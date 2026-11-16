import asyncio
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import AsyncMock, patch

from flashrag.config import Config
from flashrag.retriever import ParallelRetriever
from flashrag.utils import get_retriever


class TestParallelRetriever(unittest.TestCase):
    def setUp(self):
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = Path(directory.name)
        self.config_file = self.directory / "retriever.yaml"
        self.config_file.write_text("retrieval_method: parallel\nretrieval_topk: 2\n", encoding="utf-8")

    def get_parallel_retriever(self, **settings):
        config = Config(config_file_path=self.config_file, config_dict={"disable_save": True, **settings})
        with patch("flashrag.utils.utils.AutoConfig.from_pretrained") as load_model:
            retriever = get_retriever(config)
        load_model.assert_not_called()
        self.assertIsInstance(retriever, ParallelRetriever)
        self.assertIsNone(retriever.index_path)
        self.assertIsNone(retriever.corpus_path)
        return retriever

    def test_default_still_selects_dense_retrieval(self):
        config = Config(config_dict={"disable_save": True})
        self.assertEqual(config["retrieval_method"], "e5")
        with (
            patch("flashrag.utils.utils.AutoConfig.from_pretrained") as load_model,
            patch("flashrag.retriever.DenseRetriever") as dense,
            patch("flashrag.retriever.ParallelRetriever") as parallel,
        ):
            self.assertIs(get_retriever(config), dense.return_value)
        load_model.assert_called_once_with(config["retrieval_model_path"])
        parallel.assert_not_called()

    def test_missing_optional_sdk_reports_how_to_install_it(self):
        retriever = self.get_parallel_retriever()
        with (
            patch.dict("sys.modules", {"mcp": None}),
            self.assertRaisesRegex(ImportError, r"pip install -e '\.\[parallel\]'"),
        ):
            retriever.search("orbital period")

    def test_search_returns_pipeline_documents_and_rank_scores(self):
        retriever = self.get_parallel_retriever()
        session = AsyncMock()
        session.call_tool.return_value.isError = False
        session.call_tool.return_value.structuredContent = {
            "results": [
                {"url": "https://example.org/empty", "title": "Empty", "excerpts": []},
                {"url": "https://example.org/a", "title": "First\nsource", "excerpts": [" One ", "", "Two"]},
                {"url": "https://example.org/b", "title": None, "excerpts": ["Another passage"]},
                {"url": "https://example.org/c", "title": "Third", "excerpts": ["Outside top k"]},
            ],
        }

        async def search_queries(queries, num):
            return [await retriever._search_query(session, query, num) for query in queries]

        with patch.object(retriever, "_search_queries", side_effect=search_queries):
            documents, scores = retriever.search("orbital period", return_score=True)
            self.assertEqual(retriever.search("orbital period", num=1), documents[:1])
            self.assertEqual(retriever.batch_search(["orbital period"]), [documents])
            self.assertEqual(retriever.batch_search(["orbital period"], return_score=True), ([documents], [scores]))

        self.assertEqual(scores, [1.0, 0.5])
        self.assertEqual(
            documents[0],
            {
                "id": "https://example.org/a",
                "title": "First source",
                "text": "One\n\nTwo",
                "url": "https://example.org/a",
                "contents": "First source\nOne\n\nTwo",
            },
        )
        self.assertEqual(documents[1]["title"], documents[1]["url"])

    def test_empty_input_and_complete_cache_do_not_start_a_session(self):
        url = "https://example.org/cached"
        documents = [
            {"id": url, "title": "Cached", "text": "Text", "contents": "Cached\nText", "url": url, "score": 0.5}
        ]
        cache_path = self.directory / "retrieval_cache.json"
        cache_path.write_text(json.dumps({"cached": documents}), encoding="utf-8")
        retriever = self.get_parallel_retriever(use_retrieval_cache=True, retrieval_cache_path=str(cache_path))

        with patch.object(retriever, "_search_queries") as remote:
            self.assertEqual(retriever.search("cached", num=1, return_score=True), (documents, [0.5]))
            self.assertEqual(retriever.batch_search([], return_score=True), ([], []))
            self.assertEqual(retriever.batch_search([" ", ""], return_score=True), ([[], []], [[], []]))
            self.assertEqual(retriever.search("regular query", num=0), [])
        remote.assert_not_called()

    def test_reranker_only_runs_when_there_are_documents(self):
        with patch("flashrag.retriever.retriever.get_reranker") as load_reranker:
            retriever = self.get_parallel_retriever(use_reranker=True)
        rerank = load_reranker.return_value.rerank
        with patch.object(retriever, "_search_queries", side_effect=lambda queries, num: [[] for _ in queries]):
            self.assertEqual(retriever.search("no results", return_score=True), ([], []))
            self.assertEqual(retriever.batch_search(["blank", "empty"], return_score=True), ([[], []], [[], []]))
            self.assertEqual(retriever.batch_search([], return_score=True), ([], []))
        rerank.assert_not_called()

        query = ["empty", "matched"]
        doc = {"id": "https://example.org", "contents": "Example\nPassage"}
        rerank.return_value = ([[], [doc]], [[], [0.9]])
        with patch.object(retriever, "_search_queries", return_value=[[], [doc]]):
            self.assertEqual(retriever.batch_search(query, return_score=True), rerank.return_value)
        rerank.assert_called_once_with(query, [[], [doc]])

    def test_search_works_inside_a_running_event_loop(self):
        retriever = self.get_parallel_retriever()

        async def search():
            return retriever.batch_search(["first", "second"], return_score=True)

        with patch.object(retriever, "_search_queries", return_value=[[], [{"id": "b"}]]) as remote:
            self.assertEqual(asyncio.run(search()), ([[], [{"id": "b"}]], [[], [1.0]]))
        remote.assert_awaited_once_with(["first", "second"], 2)
