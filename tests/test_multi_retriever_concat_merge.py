"""Offline unit tests for ``MultiRetrieverRouter.post_process_result`` concat merge.

Scope: retrieval result *merging* only (``merge_method == "concat"`` dedup).

The production method under test, ``post_process_result``, is imported from the
real source file ``flashrag/retriever/retriever.py`` and is never mocked. Heavy
third-party dependencies (``faiss``/``torch``/``transformers``/...) are only
stubbed when they cannot be imported in a minimal offline environment; in a
fully provisioned CI the real module is imported directly via the ``try`` path.

Contract under test (first-seen stable dedup by doc id):
  * returned docs contain each doc id exactly once, in first-seen order;
  * the kept score for a repeated id is the score of its FIRST occurrence;
  * parallel ``score_list`` stays positionally aligned with returned docs;
  * per-query lists in the batch path are processed independently;
  * distinct docs (different id) are kept even if their contents are identical;
  * an input that already has no duplicates is returned unchanged (control).
"""

import importlib.util
import sys
import types
import unittest


def _stub(name, **attrs):
    m = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(m, k, v)
    sys.modules[name] = m
    return m


# Third-party heavy deps whose absence in a minimal offline env triggers the stub
# fallback. Any other error (e.g. a real syntax/logic error in the module under
# test) must propagate and fail the test -- it is never swallowed.
_OFFLINE_STUB_DEPS = {
    "tqdm", "faiss", "torch", "numpy", "requests", "datasets", "langid",
    "transformers",
}


def _assert_real_method(router_cls):
    import inspect
    func = router_cls.post_process_result
    assert callable(func), "post_process_result must be a real method"
    src = inspect.getsourcefile(func) or ""
    assert src.replace("\\", "/").endswith(
        "flashrag/retriever/retriever.py"
    ), f"post_process_result not loaded from real source: {src}"


def _load_real_router():
    # Preferred path: real environment / fully provisioned CI.
    try:
        from flashrag.retriever.retriever import MultiRetrieverRouter  # noqa: F401
        _assert_real_method(MultiRetrieverRouter)
        return MultiRetrieverRouter
    except ImportError as exc:
        # Fall back to stubbing ONLY when the import failure is a missing
        # third-party dependency. Anything else propagates.
        if exc.name not in _OFFLINE_STUB_DEPS:
            raise

    # Offline fallback: stub only unrelated heavy deps, then load the REAL file.
    import os

    # __file__ = <repo>/tests/test_....py  ->  repo_root = <repo>
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    real_file = os.path.join(repo_root, "flashrag", "retriever", "retriever.py")

    _stub("faiss", omp_set_num_threads=lambda *a, **k: None, read_index=lambda *a, **k: None)
    _stub("torch")
    _stub("numpy")
    _stub("tqdm", tqdm=lambda *a, **k: None)
    _stub("requests")
    _stub("datasets")
    _stub("langid")
    _stub("transformers")

    _stub("flashrag")
    _stub(
        "flashrag.utils",
        get_reranker=lambda *a, **k: None,
        get_device=lambda *a, **k: "cpu",
    )
    _stub("flashrag.retriever")
    _stub(
        "flashrag.retriever.utils",
        load_corpus=lambda *a, **k: None,
        load_docs=lambda *a, **k: None,
        convert_numpy=lambda x: x,
        judge_image=lambda *a, **k: False,
        judge_zh=lambda *a, **k: False,
    )
    _stub(
        "flashrag.retriever.encoder",
        Encoder=type("Encoder", (), {}),
        STEncoder=type("STEncoder", (), {}),
        ClipEncoder=type("ClipEncoder", (), {}),
    )

    spec = importlib.util.spec_from_file_location("flashrag.retriever.retriever_real", real_file)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["flashrag.retriever.retriever_real"] = mod
    spec.loader.exec_module(mod)
    _assert_real_method(mod.MultiRetrieverRouter)
    return mod.MultiRetrieverRouter


MultiRetrieverRouter = _load_real_router()


def mk(doc_id, contents, score=None):
    d = {"id": doc_id, "contents": contents}
    if score is not None:
        d["score"] = score
    return d


def make_router():
    r = MultiRetrieverRouter.__new__(MultiRetrieverRouter)
    r.merge_method = "concat"
    r.final_topk = 5
    return r


class TestConcatMerge(unittest.TestCase):
    def run_post(self, result_list, score_list, num=5):
        router = make_router()
        return router.post_process_result("q", result_list, score_list, num)

    def test_single_adjacent_triple_duplicate_dedups_to_first_seen(self):
        # Three adjacent docs share id "1"; only the first should survive.
        docs = [mk("1", "a"), mk("1", "a"), mk("1", "a"), mk("2", "b")]
        out_docs, _ = self.run_post(docs, [])
        ids = [d["id"] for d in out_docs]
        self.assertEqual(ids, ["1", "2"], f"expected first-seen [1,2], got {ids}")

    def test_single_repeated_id_keeps_first_score_and_aligns(self):
        # score of the repeated id must be the FIRST occurrence's score.
        # scores[1]=1.0 intentionally equals index 1 (exercises by-value remove path).
        docs = [mk("1", "a"), mk("1", "a"), mk("1", "a"), mk("2", "b")]
        scores = [10.0, 1.0, 30.0, 40.0]
        out_docs, out_scores = self.run_post(docs, scores)
        ids = [d["id"] for d in out_docs]
        self.assertEqual(ids, ["1", "2"], f"expected ids [1,2], got {ids}")
        self.assertEqual(len(out_scores), len(out_docs), "scores must align with docs")
        self.assertEqual(out_scores, [10.0, 40.0], f"expected first-seen scores [10,40], got {out_scores}")

    def test_nonadjacent_duplicate_dropped(self):
        docs = [mk("1", "a"), mk("2", "b"), mk("1", "a2"), mk("3", "c")]
        scores = [10.0, 20.0, 2.0, 40.0]  # scores[2]=2.0 == index 2
        out_docs, out_scores = self.run_post(docs, scores)
        ids = [d["id"] for d in out_docs]
        self.assertEqual(ids, ["1", "2", "3"], f"expected [1,2,3], got {ids}")
        self.assertEqual(out_scores, [10.0, 20.0, 40.0], f"got {out_scores}")

    def test_distinct_id_same_contents_both_kept(self):
        # Dedup is by id, not by content: two different ids with identical contents survive.
        docs = [mk("1", "same"), mk("2", "same"), mk("3", "x")]
        out_docs, _ = self.run_post(docs, [])
        ids = [d["id"] for d in out_docs]
        self.assertEqual(ids, ["1", "2", "3"], f"expected all kept, got {ids}")

    def test_batch_per_query_dedup_is_isolated(self):
        result_list = [
            [mk("1", "a"), mk("1", "a"), mk("1", "a"), mk("2", "b")],  # 3 adjacent dups
            [mk("9", "i"), mk("8", "h"), mk("9", "i2")],               # non-adjacent dup
        ]
        score_list = [
            [10.0, 1.0, 30.0, 40.0],  # [1]=1.0 == index 1
            [90.0, 80.0, 2.0],        # [2]=2.0 == index 2
        ]
        out_docs, out_scores = self.run_post(result_list, score_list)
        q0_ids = [d["id"] for d in out_docs[0]]
        q1_ids = [d["id"] for d in out_docs[1]]
        self.assertEqual(q0_ids, ["1", "2"], f"q0 expected [1,2], got {q0_ids}")
        self.assertEqual(q1_ids, ["9", "8"], f"q1 expected [9,8], got {q1_ids}")
        self.assertEqual(len(out_scores[0]), len(out_docs[0]))
        self.assertEqual(len(out_scores[1]), len(out_docs[1]))

    def test_control_no_duplicate_input_unchanged(self):
        docs = [mk("1", "a"), mk("2", "b"), mk("3", "c")]
        scores = [1.0, 2.0, 3.0]
        out_docs, out_scores = self.run_post(docs, scores)
        ids = [d["id"] for d in out_docs]
        self.assertEqual(ids, ["1", "2", "3"])
        self.assertEqual(out_scores, [1.0, 2.0, 3.0])

    def test_duplicate_does_not_crash_when_scores_lack_index_value(self):
        # Regression for ``score_list.remove(idx)`` by value: when no score equals
        # the dup index (1 here), the old code raised ValueError. The rebuilt path
        # must not raise and must keep the first occurrence's score.
        docs = [mk("1", "a"), mk("1", "a"), mk("1", "a"), mk("2", "b")]
        scores = [10.0, 20.0, 30.0, 40.0]  # no value == index 1
        out_docs, out_scores = self.run_post(docs, scores)  # must not raise
        ids = [d["id"] for d in out_docs]
        self.assertEqual(ids, ["1", "2"], f"expected [1,2], got {ids}")
        self.assertEqual(out_scores, [10.0, 40.0], f"expected [10,40], got {out_scores}")

    def test_single_doc_boundary_unchanged(self):
        out_docs, out_scores = self.run_post([mk("1", "only")], [5.0])
        self.assertEqual([d["id"] for d in out_docs], ["1"])
        self.assertEqual(out_scores, [5.0])


if __name__ == "__main__":
    unittest.main(verbosity=2)
