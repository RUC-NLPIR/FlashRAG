import unittest

import tiktoken

from flashrag.prompt import PromptTemplate


class TestCompletionPromptTruncation(unittest.TestCase):
    def setUp(self):
        self.template = PromptTemplate(
            {"framework": "openai", "generator_max_input_len": 12},
            system_prompt="",
            user_prompt="{question}",
            enable_chat=False,
        )
        self.template.tokenizer = tiktoken.Encoding(
            name="test_bytes",
            pat_str=r"[\s\S]",
            mergeable_ranks={bytes([i]): i for i in range(256)},
            special_tokens={},
        )

    def test_short_prompt_is_not_duplicated(self):
        self.assertEqual(self.template.truncate_prompt("hello"), "hello")

    def test_prompt_between_half_and_full_budget_is_unchanged(self):
        self.assertEqual(self.template.truncate_prompt("abcdefgh"), "abcdefgh")

    def test_prompt_at_token_limit_is_unchanged(self):
        self.assertEqual(self.template.truncate_prompt("abcdefghijkl"), "abcdefghijkl")

    def test_empty_prompt_is_unchanged(self):
        self.assertEqual(self.template.truncate_prompt(""), "")

    def test_long_prompt_preserves_head_and_tail_truncation(self):
        self.assertEqual(
            self.template.truncate_prompt("abcdefghijklmnop"), "abcdefklmnop"
        )

    def test_get_string_preserves_formatted_completion_prompt(self):
        self.assertEqual(self.template.get_string(question="hello"), "hello")

    def test_get_string_preserves_string_messages(self):
        self.assertEqual(self.template.get_string(messages="hello"), "hello")


if __name__ == "__main__":
    unittest.main()
