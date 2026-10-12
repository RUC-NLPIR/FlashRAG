import math
import unittest
from types import SimpleNamespace

from flashrag.evaluator.metrics import (
    BLEU,
    Rouge_1,
    Rouge_2,
    Rouge_L,
    ZH_Rouge_1,
    ZH_Rouge_2,
    ZH_Rouge_L,
)


def build_data(preds, golden_answers):
    return SimpleNamespace(
        pred=preds,
        golden_answers=golden_answers,
        choices=[[] for _ in preds],
    )


class TestRougeMetrics(unittest.TestCase):
    def test_empty_prediction_scores_zero(self):
        data = build_data(
            ["the capital is paris", "", "   "],
            [["the capital is paris"], ["rome"], ["rome"]],
        )

        for metric_class in (Rouge_1, Rouge_2, Rouge_L):
            with self.subTest(metric=metric_class.metric_name):
                result, sample_scores = metric_class(
                    {"dataset_name": "nq"}
                ).calculate_metric(data)
                self.assertAlmostEqual(sample_scores[0], 1.0, places=5)
                self.assertEqual(sample_scores[1:], [0.0, 0.0])
                self.assertAlmostEqual(
                    result[metric_class.metric_name], 1 / 3, places=5
                )

    def test_empty_golden_answer_does_not_hide_other_answers(self):
        data = build_data(["rome"], [["", "rome"]])

        result, sample_scores = Rouge_1({"dataset_name": "nq"}).calculate_metric(data)

        self.assertAlmostEqual(sample_scores[0], 1.0, places=5)
        self.assertAlmostEqual(result["rouge-1"], 1.0, places=5)

    def test_zh_empty_prediction_scores_zero(self):
        data = build_data(["北京是中国的首都", ""], [["北京是中国的首都"], ["上海"]])

        for metric_class in (ZH_Rouge_1, ZH_Rouge_2, ZH_Rouge_L):
            with self.subTest(metric=metric_class.metric_name):
                _, sample_scores = metric_class(
                    {"dataset_name": "zh"}
                ).calculate_metric(data)
                self.assertAlmostEqual(sample_scores[0], 1.0, places=5)
                self.assertEqual(sample_scores[1], 0.0)


class TestBleu(unittest.TestCase):
    def test_empty_prediction_scores_zero(self):
        data = build_data(
            ["the capital of france is paris", ""],
            [["the capital of france is paris"], ["rome"]],
        )

        for smooth in (False, True):
            with self.subTest(smooth=smooth):
                config = {"dataset_name": "nq", "metric_setting": {"bleu_smooth": smooth}}
                result, sample_scores = BLEU(config).calculate_metric(data)
                self.assertAlmostEqual(sample_scores[0], 1.0)
                self.assertEqual(sample_scores[1], 0.0)
                # 6 predicted tokens against 7 reference tokens over the corpus
                self.assertAlmostEqual(result["bleu"], math.exp(1 - 7 / 6))


if __name__ == "__main__":
    unittest.main()
