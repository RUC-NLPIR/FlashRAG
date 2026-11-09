import unittest

from flashrag.prompt.base_prompt import PromptTemplate


class WhitespaceTokenizer:
    def encode(self, text):
        return text.split()

    def decode(self, tokens):
        return " ".join(tokens)


class TestPromptTruncation(unittest.TestCase):
    def test_chat_truncation_does_not_mutate_messages(self):
        template = PromptTemplate(
            {
                "framework": "openai",
                "generator_max_input_len": 1,
                "generator_model": "unused",
            }
        )
        template.tokenizer = WhitespaceTokenizer()
        messages = [{"role": "user", "content": "alpha beta"}]

        truncated = template.get_string(messages=messages)

        self.assertEqual(
            truncated,
            [{"role": "user", "content": "alpha"}],
        )
        self.assertEqual(messages, [{"role": "user", "content": "alpha beta"}])
