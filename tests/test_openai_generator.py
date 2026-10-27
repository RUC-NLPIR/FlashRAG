import asyncio
import contextlib
import io
import unittest
from types import SimpleNamespace

from flashrag.generator.openai_generator import OpenaiGenerator


class FakeCompletions:
    def __init__(self, choice):
        self.choice = choice
        self.requests = []

    async def create(self, **kwargs):
        self.requests.append(kwargs)
        return SimpleNamespace(choices=[self.choice])


class TestOpenaiGenerator(unittest.TestCase):
    def test_get_response_does_not_write_prompts_to_stdout(self):
        chat_completions = FakeCompletions(
            SimpleNamespace(message=SimpleNamespace(content="chat response"))
        )
        completions = FakeCompletions(SimpleNamespace(text="completion response"))
        generator = OpenaiGenerator.__new__(OpenaiGenerator)
        generator.model_name = "test-model"
        generator.client = SimpleNamespace(
            chat=SimpleNamespace(completions=chat_completions),
            completions=completions,
        )
        messages = [{"role": "user", "content": "private prompt"}]

        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            chat_choice = asyncio.run(generator._get_response(messages, temperature=0))
            completion_choice = asyncio.run(
                generator._get_response("private prompt", mode="completion")
            )

        self.assertEqual(stdout.getvalue(), "")
        self.assertEqual(chat_choice.message.content, "chat response")
        self.assertEqual(completion_choice.text, "completion response")
        self.assertEqual(
            chat_completions.requests,
            [
                {
                    "model": "test-model",
                    "messages": messages,
                    "temperature": 0,
                }
            ],
        )
        self.assertEqual(
            completions.requests,
            [{"model": "test-model", "prompt": "private prompt"}],
        )


if __name__ == "__main__":
    unittest.main()
