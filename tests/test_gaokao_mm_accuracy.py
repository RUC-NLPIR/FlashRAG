import unittest

from flashrag.dataset import Dataset, Item
from flashrag.evaluator.metrics import GAOKAOMM_Accuracy
from flashrag.utils import gaokaomm_pred_parse


def make_dataset(samples):
    items = []
    for index, (question_type, golden_answers, model_output) in enumerate(samples):
        item = Item(
            {
                "id": str(index),
                "question": f"Question {index}",
                "golden_answers": golden_answers,
                "question_type": question_type,
                "subject": "Physics",
            }
        )
        item.update_output("pred", model_output)
        items.append(item)
    return Dataset(config={"dataset_name": "gaokao_mm"}, data=items)


class TestGAOKAOMMAccuracy(unittest.TestCase):
    def setUp(self):
        self.metric = GAOKAOMM_Accuracy({"dataset_name": "gaokao_mm"})

    def score(self, samples):
        dataset = gaokaomm_pred_parse(make_dataset(samples))
        return self.metric.calculate_metric(dataset)

    def test_unparsable_multiple_choice_answer_scores_zero(self):
        metric_dict, scores = self.score(
            [
                ("multiple_choice", ["A", "C"], "【解析】不确定。<eoe>\n【答案】 <eoa>"),
                ("multiple_choice", ["B", "D"], "I cannot answer this question."),
            ]
        )

        self.assertEqual(scores, [0.0, 0.0])
        self.assertEqual(metric_dict["avg_score"], 0.0)

    def test_multiple_choice_scoring(self):
        _, scores = self.score(
            [
                ("multiple_choice", ["A", "C"], "【答案】 AC <eoa>"),
                ("multiple_choice", ["A", "C"], "【答案】 A <eoa>"),
                ("multiple_choice", ["A", "C"], "【答案】 AB <eoa>"),
            ]
        )

        self.assertEqual(scores, [1.0, 0.5, 0.0])

    def test_single_choice_scoring(self):
        _, scores = self.score(
            [
                ("single_choice", ["B"], "【答案】 B <eoa>"),
                ("single_choice", ["B"], "【答案】 C <eoa>"),
                ("single_choice", ["B"], "no idea"),
            ]
        )

        self.assertEqual(scores, [1.0, 0.0, 0.0])


if __name__ == "__main__":
    unittest.main()
