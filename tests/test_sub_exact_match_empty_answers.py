import unittest

from flashrag.evaluator.metrics import Sub_ExactMatch


class TestSubExactMatchEmptyAnswers(unittest.TestCase):
    def test_empty_answer_only_matches_empty_prediction(self):
        metric = Sub_ExactMatch({"dataset_name": "nq"})

        self.assertEqual(metric.calculate_sub_em("unrelated answer", [""]), 0.0)
        self.assertEqual(metric.calculate_sub_em("", [""]), 1.0)

    def test_empty_answer_does_not_hide_other_answers(self):
        metric = Sub_ExactMatch({"dataset_name": "nq"})

        self.assertEqual(metric.calculate_sub_em("unrelated answer", ["", "rome"]), 0.0)
        self.assertEqual(
            metric.calculate_sub_em("the answer is Rome", ["", "rome"]), 1.0
        )

    def test_empty_regex_answer_only_matches_empty_prediction(self):
        metric = Sub_ExactMatch({"dataset_name": "curatedtrec"})

        self.assertEqual(metric.calculate_sub_em("unrelated answer", [""]), 0.0)
        self.assertEqual(metric.calculate_sub_em("", [""]), 1.0)


if __name__ == "__main__":
    unittest.main()
