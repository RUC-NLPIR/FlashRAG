import unittest
from types import SimpleNamespace

from flashrag.evaluator.evaluator import Evaluator
from flashrag.evaluator.metrics import (
    DocumentRetrievalF1,
    DocumentRetrievalMAP,
    DocumentRetrievalPrecision,
    DocumentRetrievalRecall,
)


def build_config(topk=3, **metric_setting):
    return {
        "dataset_name": "document_retrieval_test",
        "metric_setting": {
            "retrieval_recall_topk": topk,
            **metric_setting,
        },
    }


class TestDocumentRetrievalMetrics(unittest.TestCase):
    def setUp(self):
        self.metrics = [
            (DocumentRetrievalRecall, 1.0),
            (DocumentRetrievalPrecision, 2 / 3),
            (DocumentRetrievalF1, 0.8),
            (DocumentRetrievalMAP, 5 / 6),
        ]

    def test_scores_ranked_unique_parent_documents(self):
        data = [
            SimpleNamespace(
                golden_doc_ids=["doc-1", "doc-2"],
                retrieval_result=[
                    {"id": "chunk-1", "doc_id": "doc-1"},
                    {"id": "chunk-2", "doc_id": "doc-1"},
                    {"id": "chunk-3", "doc_id": "doc-3"},
                    {"id": "chunk-4", "doc_id": "doc-2"},
                ],
            )
        ]

        for metric_class, expected in self.metrics:
            with self.subTest(metric=metric_class.metric_name):
                result, sample_scores = metric_class(build_config()).calculate_metric(
                    data
                )
                self.assertAlmostEqual(sample_scores[0], expected)
                self.assertAlmostEqual(
                    result[f"{metric_class.metric_name}_top3"], expected
                )

    def test_falls_back_to_id_and_normalizes_id_types(self):
        data = [
            SimpleNamespace(
                golden_doc_ids=["7"],
                retrieval_result=[{"id": 7, "contents": "matched document"}],
            )
        ]

        result, sample_scores = DocumentRetrievalRecall(
            build_config(topk=1)
        ).calculate_metric(data)

        self.assertEqual(result, {"retrieval_doc_recall_top1": 1.0})
        self.assertEqual(sample_scores, [1.0])

    def test_uses_configured_document_id_fields(self):
        data = [
            SimpleNamespace(
                relevant_sources=["source-a"],
                retrieval_result=[{"source_id": "source-a"}],
            )
        ]
        config = build_config(
            topk=1,
            golden_document_id_field="relevant_sources",
            document_id_field="source_id",
        )

        _, sample_scores = DocumentRetrievalRecall(config).calculate_metric(data)

        self.assertEqual(sample_scores, [1.0])

    def test_empty_retrieval_scores_zero(self):
        data = [
            SimpleNamespace(
                golden_doc_ids=["doc-1"],
                retrieval_result=[],
            )
        ]

        for metric_class, _ in self.metrics:
            with self.subTest(metric=metric_class.metric_name):
                _, sample_scores = metric_class(build_config()).calculate_metric(data)
                self.assertEqual(sample_scores, [0.0])

    def test_precision_uses_k_when_fewer_documents_are_returned(self):
        data = [
            SimpleNamespace(
                golden_doc_ids=["doc-1"],
                retrieval_result=[{"id": "doc-1"}],
            )
        ]

        _, precision_scores = DocumentRetrievalPrecision(
            build_config(topk=3)
        ).calculate_metric(data)
        _, f1_scores = DocumentRetrievalF1(build_config(topk=3)).calculate_metric(data)

        self.assertAlmostEqual(precision_scores[0], 1 / 3)
        self.assertAlmostEqual(f1_scores[0], 0.5)

    def test_rejects_missing_or_empty_ground_truth(self):
        cases = [
            SimpleNamespace(retrieval_result=[]),
            SimpleNamespace(golden_doc_ids=[], retrieval_result=[]),
        ]

        for item in cases:
            with self.subTest(item=item):
                with self.assertRaisesRegex(ValueError, "Sample 0"):
                    DocumentRetrievalRecall(build_config()).calculate_metric([item])

    def test_rejects_missing_retrieved_document_id(self):
        data = [
            SimpleNamespace(
                golden_doc_ids=["doc-1"],
                retrieval_result=[{"contents": "no identifier"}],
            )
        ]

        with self.assertRaisesRegex(ValueError, "document ID field `id`"):
            DocumentRetrievalRecall(build_config()).calculate_metric(data)

    def test_rejects_non_positive_topk(self):
        with self.assertRaisesRegex(ValueError, "greater than 0"):
            DocumentRetrievalRecall(build_config(topk=0))

    def test_metrics_are_discoverable_by_evaluator(self):
        available_metrics = Evaluator.__new__(Evaluator)._collect_metrics()

        for metric_class, _ in self.metrics:
            self.assertIs(available_metrics[metric_class.metric_name], metric_class)


if __name__ == "__main__":
    unittest.main()
