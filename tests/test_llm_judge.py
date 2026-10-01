import unittest
from types import SimpleNamespace

from flashrag.evaluator.metrics import LLMJudge


class FakePipeline:
    def __init__(self, ratings):
        self.ratings = ratings
        self.prompts = None

    def __call__(self, prompts, max_new_tokens, batch_size):
        self.prompts = prompts
        assert max_new_tokens == 100
        assert batch_size == 8
        return [
            {"generated_text": f"Feedback:::\nTotal rating: {rating}"}
            for rating in self.ratings
        ]


class TestLLMJudge(unittest.TestCase):
    def test_extract_judge_score_as_instance_method(self):
        judge = LLMJudge.__new__(LLMJudge)

        score = judge.extract_judge_score(
            "Feedback:::\nTotal rating: 7.5\nAdditional feedback"
        )

        self.assertEqual(score, 7.5)

    def test_calculate_metric_normalizes_scores_to_zero_one_range(self):
        judge = LLMJudge.__new__(LLMJudge)
        judge.llm_pipeline = FakePipeline([0, 7.5, 10])
        data = SimpleNamespace(
            question=["q0", "q1", "q2"],
            pred=["a0", "a1", "a2"],
        )

        metric, sample_scores = judge.calculate_metric(data)

        self.assertEqual(sample_scores, [0.0, 0.75, 1.0])
        self.assertAlmostEqual(metric["llm_judge_score"], 7 / 12)
        self.assertEqual(len(judge.llm_pipeline.prompts), 3)
        self.assertIn("Question: q1", judge.llm_pipeline.prompts[1])
        self.assertIn("Answer: a1", judge.llm_pipeline.prompts[1])


if __name__ == "__main__":
    unittest.main()
