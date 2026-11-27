import unittest
from types import SimpleNamespace

from flashrag.evaluator.metrics import Retrieval_Precision, Retrieval_Recall


CONFIG = {
    "dataset_name": "empty_golden_answer_retrieval_test",
    "metric_setting": {"retrieval_recall_topk": 2},
}


def build_data(golden_answers, documents):
    return SimpleNamespace(
        choices=[[] for _ in golden_answers],
        golden_answers=golden_answers,
        retrieval_result=documents,
    )


def contents(*texts):
    return [{"contents": text} for text in texts]


class TestEmptyGoldenAnswerRetrievalMetrics(unittest.TestCase):
    def test_empty_golden_answer_does_not_match_unrelated_documents(self):
        data = build_data(
            [[""]],
            [contents("completely unrelated document", "another unrelated document")],
        )

        for metric_class in (Retrieval_Recall, Retrieval_Precision):
            with self.subTest(metric=metric_class.metric_name):
                _, scores = metric_class(CONFIG).calculate_metric(data)
                self.assertEqual(scores, [0.0])

    def test_empty_golden_answer_does_not_hide_other_answers(self):
        data = build_data(
            [["", "Paris"], ["", "Paris"]],
            [contents("London", "Rome"), contents("Paris", "London")],
        )

        recall = Retrieval_Recall(CONFIG).calculate_metric(data)[1]
        precision = Retrieval_Precision(CONFIG).calculate_metric(data)[1]
        self.assertEqual(recall, [0.0, 1.0])
        self.assertEqual(precision, [0.0, 0.5])

    def test_empty_document_still_matches_empty_golden_answer(self):
        data = build_data([[""]], [contents("", "")])

        for metric_class in (Retrieval_Recall, Retrieval_Precision):
            with self.subTest(metric=metric_class.metric_name):
                _, scores = metric_class(CONFIG).calculate_metric(data)
                self.assertEqual(scores, [1.0])

    def test_nonempty_golden_answers_are_unchanged(self):
        data = build_data(
            [["Paris"], ["Paris"]],
            [contents("Paris", "London"), contents("London", "Rome")],
        )

        for metric_class in (Retrieval_Recall, Retrieval_Precision):
            with self.subTest(metric=metric_class.metric_name):
                _, scores = metric_class(CONFIG).calculate_metric(data)
                if metric_class is Retrieval_Recall:
                    self.assertEqual(scores, [1.0, 0.0])
                else:
                    self.assertEqual(scores, [0.5, 0.0])


if __name__ == "__main__":
    unittest.main()
