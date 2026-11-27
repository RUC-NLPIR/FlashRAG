import unittest
from types import SimpleNamespace

from flashrag.evaluator.metrics import ExactMatch, Sub_ExactMatch


class TestRegexAnswerMetrics(unittest.TestCase):
    def setUp(self):
        self.exact = ExactMatch({"dataset_name": "curatedtrec"})
        self.substring = Sub_ExactMatch({"dataset_name": "curatedtrec"})

    def test_regex_preserves_articles_and_punctuation(self):
        # These answer texts occur in the official CuratedTREC dataset.
        cases = [
            ("After the Fall", r"After the Fall|Finishing the Play"),
            ("The Star-Spangled Banner", r"The Star-Spangled Banner"),
            ("The Ameraucana", r"The Ameraucana"),
            ("3.14", r"3\.14"),
            ("C++", r"C\+\+"),
        ]
        for prediction, pattern in cases:
            with self.subTest(prediction=prediction):
                self.assertEqual(self.exact.calculate_em(prediction, [pattern]), 1.0)
                self.assertEqual(
                    self.substring.calculate_sub_em(prediction, [pattern]), 1.0
                )

    def test_regex_remains_case_insensitive_and_trims_whitespace(self):
        prediction = "  after THE fall\n"
        pattern = r"After the Fall"
        self.assertEqual(self.exact.calculate_em(prediction, [pattern]), 1.0)
        self.assertEqual(self.substring.calculate_sub_em(prediction, [pattern]), 1.0)

    def test_regex_keeps_exact_and_substring_matching_distinct(self):
        pattern = r"After the Fall"
        prediction = "The answer is After the Fall."
        self.assertEqual(self.exact.calculate_em(prediction, [pattern]), 0.0)
        self.assertEqual(self.substring.calculate_sub_em(prediction, [pattern]), 1.0)
        self.assertEqual(self.exact.calculate_em("After the Rain", [pattern]), 0.0)
        self.assertEqual(
            self.substring.calculate_sub_em("After the Rain", [pattern]), 0.0
        )

    def test_calculate_metric_keeps_regex_answers(self):
        data = SimpleNamespace(
            pred=["After the Fall", "The Star-Spangled Banner"],
            golden_answers=[[r"After the Fall"], [r"The Star-Spangled Banner"]],
            choices=[[], []],
        )
        self.assertEqual(self.exact.calculate_metric(data), ({"em": 1.0}, [1.0, 1.0]))
        self.assertEqual(
            self.substring.calculate_metric(data), ({"acc": 1.0}, [1.0, 1.0])
        )

    def test_plain_text_datasets_still_normalize_predictions(self):
        prediction = " The PARIS! "
        self.assertEqual(
            ExactMatch({"dataset_name": "nq"}).calculate_em(prediction, ["paris"]), 1.0
        )
        self.assertEqual(
            Sub_ExactMatch({"dataset_name": "nq"}).calculate_sub_em(
                prediction, ["paris"]
            ),
            1.0,
        )

    def test_empty_regex_does_not_hide_valid_references(self):
        self.assertEqual(self.substring.calculate_sub_em("  ", [""]), 1.0)
        self.assertEqual(self.substring.calculate_sub_em("unrelated", [""]), 0.0)
        self.assertEqual(
            self.substring.calculate_sub_em("After the Fall", ["", r"After the Fall"]),
            1.0,
        )


if __name__ == "__main__":
    unittest.main()
