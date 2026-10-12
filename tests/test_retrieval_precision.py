import unittest
from types import SimpleNamespace

from flashrag.evaluator.metrics import Retrieval_Precision


class TestRetrievalPrecision(unittest.TestCase):
    def setUp(self):
        self.metric = Retrieval_Precision({
            "dataset_name": "retrieval_precision_test",
            "metric_setting": {"retrieval_recall_topk": 2},
        })

    def score(self, results):
        data = SimpleNamespace(
            choices=[[] for _ in results],
            golden_answers=[["Paris"] for _ in results],
            retrieval_result=results,
        )
        return self.metric.calculate_metric(data)

    def test_empty_retrieval_scores_zero(self):
        with self.assertWarnsRegex(UserWarning, "smaller than topk"):
            result, scores = self.score([[]])
        self.assertEqual(scores, [0.0])
        self.assertEqual(result, {"retrieval_precision_top2": 0.0})

    def test_empty_sample_is_included_in_batch_average(self):
        with self.assertWarnsRegex(UserWarning, "smaller than topk"):
            result, scores = self.score([
                [],
                [{"contents": "Paris"}, {"contents": "London"}],
            ])
        self.assertEqual(scores, [0.0, 0.5])
        self.assertEqual(result, {"retrieval_precision_top2": 0.25})

    def test_short_nonempty_retrieval_keeps_existing_denominator(self):
        with self.assertWarnsRegex(UserWarning, "smaller than topk"):
            result, scores = self.score([[{"contents": "Paris"}]])
        self.assertEqual(scores, [1.0])
        self.assertEqual(result, {"retrieval_precision_top2": 1.0})

    def test_only_topk_documents_are_scored(self):
        result, scores = self.score([[
            {"contents": "Paris"},
            {"contents": "London"},
            {"contents": "Paris"},
        ]])
        self.assertEqual(scores, [0.5])
        self.assertEqual(result, {"retrieval_precision_top2": 0.5})


if __name__ == "__main__":
    unittest.main()
