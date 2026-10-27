from unittest.mock import Mock, patch

from flashrag.retriever.retriever import MultiRetrieverRouter


def test_rerank_does_not_print_retrieved_documents():
    router = object.__new__(MultiRetrieverRouter)
    router.merge_method = "rerank"
    router.reranker = Mock()

    results = [[{"id": "doc-1", "contents": "private retrieved passage", "is_multimodal": False}]]
    scores = [[0.9]]
    router.reranker.rerank.return_value = (results, scores)

    with patch("builtins.print") as print_mock:
        actual_results, actual_scores = router.post_process_result(
            query=["question"], result_list=results, score_list=scores, num=1
        )

    print_mock.assert_not_called()
    assert actual_results == results
    assert actual_scores == scores
