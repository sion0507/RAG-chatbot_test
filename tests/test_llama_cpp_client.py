from __future__ import annotations

import unittest

from src.llm.llama_cpp_client import LlamaCppClient


class LlamaCppClientTest(unittest.TestCase):
    def test_generate_parses_json_payload(self) -> None:
        def fake_completion(**kwargs):
            _ = kwargs
            return {
                "choices": [
                    {
                        "text": '{"answer":"근거 기반 답변","needs_abstain":false,"reason":""}'
                    }
                ]
            }

        client = LlamaCppClient(
            gguf_path="unused.gguf",
            n_ctx=1024,
            temperature=0.2,
            top_p=0.9,
            max_tokens=256,
            repeat_penalty=1.1,
            stop=["</s>"],
            create_completion=fake_completion,
        )

        payload = client.generate(system_prompt="sys", user_prompt="user")
        self.assertEqual(payload["answer"], "근거 기반 답변")
        self.assertFalse(payload["needs_abstain"])

    def test_generate_falls_back_to_abstain_when_non_json(self) -> None:
        def fake_completion(**kwargs):
            _ = kwargs
            return {"choices": [{"text": "일반 텍스트 응답"}]}

        client = LlamaCppClient(
            gguf_path="unused.gguf",
            n_ctx=1024,
            temperature=0.2,
            top_p=0.9,
            max_tokens=256,
            repeat_penalty=1.1,
            stop=["</s>"],
            create_completion=fake_completion,
        )

        payload = client.generate(system_prompt="sys", user_prompt="user")
        self.assertEqual(payload["answer"], "")
        self.assertTrue(payload["needs_abstain"])
        self.assertEqual(payload["reason"], "llm_output_not_json")

    def test_generate_extracts_first_json_from_chatty_output(self) -> None:
        def fake_completion(**kwargs):
            _ = kwargs
            return {
                "choices": [
                    {
                        "text": (
                            '설명 텍스트\n'
                            '{"answer":"부분 답변","needs_abstain":false,"reason":"draft"}\n'
                            '<SYSTEM>\n'
                            '{"answer":"","needs_abstain":true,"reason":"insufficient_context"}'
                        )
                    }
                ]
            }

        client = LlamaCppClient(
            gguf_path="unused.gguf",
            n_ctx=1024,
            temperature=0.2,
            top_p=0.9,
            max_tokens=256,
            repeat_penalty=1.1,
            stop=["</s>"],
            create_completion=fake_completion,
        )

        payload = client.generate(system_prompt="sys", user_prompt="user")
        self.assertEqual(payload["answer"], "부분 답변")
        self.assertFalse(payload["needs_abstain"])
        self.assertEqual(payload["reason"], "draft")


if __name__ == "__main__":
    unittest.main()
