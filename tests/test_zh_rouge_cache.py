import unittest
from unittest.mock import patch

from flashrag.evaluator.metrics import ZH_Rouge_1, ZH_Rouge_2, ZH_Rouge_L, ZH_Rouge_Score


class TestZhRougeScoreCache(unittest.TestCase):
    def setUp(self):
        # cached_scores is a class attribute shared by all ZH_Rouge_* classes;
        # clear it so each test starts from a cold cache.
        ZH_Rouge_Score.cached_scores.clear()

    def test_repeated_call_reuses_cache(self):
        metric = ZH_Rouge_1({"dataset_name": "zh"})
        pred = "北京是中国的首都"
        golden_answers = [pred]

        with patch.object(
            metric.scorer, "get_scores", wraps=metric.scorer.get_scores
        ) as scorer:
            first = metric.calculate_rouge(pred, golden_answers)
            second = metric.calculate_rouge(pred, golden_answers)
            self.assertEqual(scorer.call_count, 1)
            self.assertEqual(first, second)

    def test_cache_key_uses_original_prediction(self):
        metric = ZH_Rouge_1({"dataset_name": "zh"})
        pred = "北京是中国的首都"
        golden_answers = [pred]

        metric.calculate_rouge(pred, golden_answers)

        self.assertEqual(
            list(ZH_Rouge_Score.cached_scores.keys()),
            [(pred, tuple(golden_answers))],
        )

    def test_cache_shared_across_zh_rouge_variants(self):
        # PR #111 introduced the cache so that rouge-1/2/l over the same
        # prediction reuse one scoring pass.
        metrics = [
            ZH_Rouge_1({"dataset_name": "zh"}),
            ZH_Rouge_2({"dataset_name": "zh"}),
            ZH_Rouge_L({"dataset_name": "zh"}),
        ]
        pred = "北京是中国的首都"
        golden_answers = [pred]

        with patch.object(
            metrics[0].scorer,
            "get_scores",
            wraps=metrics[0].scorer.get_scores,
        ) as first_scorer, patch.object(
            metrics[1].scorer,
            "get_scores",
            wraps=metrics[1].scorer.get_scores,
        ) as second_scorer, patch.object(
            metrics[2].scorer,
            "get_scores",
            wraps=metrics[2].scorer.get_scores,
        ) as third_scorer:
            scores = [m.calculate_rouge(pred, golden_answers) for m in metrics]
            self.assertEqual(first_scorer.call_count, 1)
            self.assertEqual(second_scorer.call_count, 0)
            self.assertEqual(third_scorer.call_count, 0)
            self.assertEqual(scores[0], scores[1])
            self.assertEqual(scores[1], scores[2])

    def test_cached_result_matches_fresh_computation(self):
        metric = ZH_Rouge_1({"dataset_name": "zh"})
        pred = "北京是中国的首都"
        golden_answers = ["上海", "北京"]

        cached = metric.calculate_rouge(pred, golden_answers)
        ZH_Rouge_Score.cached_scores.clear()
        fresh = metric.calculate_rouge(pred, golden_answers)

        self.assertEqual(cached, fresh)


if __name__ == "__main__":
    unittest.main()
