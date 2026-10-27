import copy
import unittest

from flashrag.dataset import Dataset
from flashrag.dataset.utils import get_batch_dataset
from flashrag.evaluator import Evaluator
from flashrag.evaluator.metrics import BaseMetric

METRICS = ["em", "f1", "acc", "precision", "recall"]
CONFIG = {
    "dataset_name": "mixed_choice_test",
    "save_dir": ".",
    "save_metric_score": False,
    "save_intermediate_data": False,
    "metrics": METRICS,
    "metric_setting": {},
}
CHOICE = {
    "id": "choice",
    "question": "Choose the sky color.",
    "choices": ["blue sky", "gray cloud"],
    "golden_answers": [0],
    "output": {"pred": "blue sky"},
}
FREE = {
    "id": "free",
    "question": "Describe the meadow.",
    "golden_answers": ["green meadow", "green field"],
    "output": {"pred": "purple stone"},
}
MULTIPLE_ANSWERS = {
    "id": "multiple_answers",
    "question": "Choose a valid description.",
    "choices": ["red brick", "gray cloud", "white snow"],
    "golden_answers": [0, 2],
    "output": {"pred": "white snow"},
}


def make_dataset(records):
    return Dataset(config=CONFIG, data=copy.deepcopy(records))


class TestMixedChoiceMetrics(unittest.TestCase):
    def assert_evaluation(self, dataset, expected_scores):
        result = Evaluator(CONFIG).evaluate(dataset)
        self.assertEqual(set(result), set(METRICS))
        for metric in METRICS:
            self.assertAlmostEqual(
                result[metric], sum(expected_scores) / len(expected_scores)
            )
            self.assertEqual(
                [item.output["metric_score"][metric] for item in dataset],
                expected_scores,
            )
        return result

    def test_resolves_each_rows_choices_and_preserves_answer_lists(self):
        explicit_empty_choices = {
            "id": "explicit_empty",
            "choices": [],
            "golden_answers": ["first alias", "second alias"],
        }
        dataset = make_dataset([CHOICE, FREE, MULTIPLE_ANSWERS, explicit_empty_choices])
        original_answers = copy.deepcopy(dataset.golden_answers)
        original_choices = copy.deepcopy(dataset.choices)

        answers = BaseMetric(CONFIG).get_dataset_answer(dataset)

        self.assertEqual(
            answers,
            [
                ["blue sky"],
                ["green meadow", "green field"],
                ["red brick", "white snow"],
                ["first alias", "second alias"],
            ],
        )
        self.assertEqual(dataset.golden_answers, original_answers)
        self.assertEqual(dataset.choices, original_choices)

    def test_mixed_rows_keep_per_item_scores_and_full_denominator(self):
        self.assert_evaluation(
            make_dataset([CHOICE, FREE, MULTIPLE_ANSWERS]), [1.0, 0.0, 1.0]
        )

    def test_free_answer_before_choice_rows_keeps_scores(self):
        self.assert_evaluation(
            make_dataset([FREE, MULTIPLE_ANSWERS, CHOICE]), [0.0, 1.0, 1.0]
        )

    def test_batch_boundaries_preserve_sample_and_aggregate_scores(self):
        expected_by_id = {"choice": 1.0, "free": 0.0, "multiple_answers": 1.0}
        for batch_size in [1, 2]:
            dataset = make_dataset([CHOICE, FREE, MULTIPLE_ANSWERS])
            totals = dict.fromkeys(METRICS, 0.0)
            count = 0
            for batch in get_batch_dataset(dataset, batch_size=batch_size):
                expected = [expected_by_id[item.id] for item in batch]
                result = self.assert_evaluation(batch, expected)
                count += len(batch)
                for metric in METRICS:
                    totals[metric] += result[metric] * len(batch)
            self.assertEqual(count, 3)
            for metric in METRICS:
                self.assertAlmostEqual(totals[metric] / count, 2 / 3)

    def test_choice_only_rows_preserve_multiple_correct_answers(self):
        self.assert_evaluation(make_dataset([CHOICE, MULTIPLE_ANSWERS]), [1.0, 1.0])

    def test_free_answer_only_rows_preserve_aliases(self):
        alias = {
            "id": "alias",
            "golden_answers": ["blue sky", "clear sky"],
            "output": {"pred": "clear sky"},
        }
        self.assert_evaluation(make_dataset([FREE, alias]), [0.0, 1.0])

    def test_empty_dataset_has_no_resolved_answers(self):
        self.assertEqual(BaseMetric(CONFIG).get_dataset_answer(make_dataset([])), [])

    def test_invalid_choice_index_is_not_silently_accepted(self):
        invalid = copy.deepcopy(CHOICE)
        invalid["golden_answers"] = [99]
        with self.assertRaises(IndexError):
            BaseMetric(CONFIG).get_dataset_answer(make_dataset([invalid]))

    def test_answer_based_retrieval_uses_each_rows_answer_text(self):
        dataset = make_dataset([CHOICE, FREE, MULTIPLE_ANSWERS])
        dataset.update_output(
            "retrieval_result",
            [
                [{"contents": "blue sky"}, {"contents": "gray cloud"}],
                [{"contents": "purple stone"}, {"contents": "gray cloud"}],
                [{"contents": "white snow"}, {"contents": "red brick"}],
            ],
        )
        config = {
            **CONFIG,
            "metrics": ["retrieval_recall", "retrieval_precision"],
            "metric_setting": {"retrieval_recall_topk": 2},
        }

        result = Evaluator(config).evaluate(dataset)

        self.assertEqual(
            result,
            {"retrieval_recall_top2": 2 / 3, "retrieval_precision_top2": 0.5},
        )
        for metric, expected in [
            ("retrieval_recall", [1, 0, 1]),
            ("retrieval_precision", [0.5, 0.0, 1.0]),
        ]:
            self.assertEqual(
                [item.output["metric_score"][metric] for item in dataset], expected
            )


if __name__ == "__main__":
    unittest.main()
