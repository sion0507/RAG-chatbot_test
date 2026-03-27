"""llama.cpp client wrapper for grounded answer generation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable


class LlamaCppClient:
    """Thin adapter over llama_cpp.Llama completion API."""

    def __init__(
        self,
        *,
        gguf_path: str,
        n_ctx: int,
        temperature: float,
        top_p: float,
        max_tokens: int,
        repeat_penalty: float,
        stop: list[str] | None,
        create_completion: Callable[..., dict[str, Any]] | None = None,
    ) -> None:
        if n_ctx <= 0:
            raise ValueError("n_ctx must be a positive integer.")
        if max_tokens <= 0:
            raise ValueError("max_tokens must be a positive integer.")

        self._temperature = temperature
        self._top_p = top_p
        self._max_tokens = max_tokens
        self._repeat_penalty = repeat_penalty
        self._stop = stop or []

        if create_completion is not None:
            self._create_completion = create_completion
            return

        model_path = Path(gguf_path)
        if not model_path.exists():
            raise FileNotFoundError(f"LLM gguf 파일을 찾을 수 없습니다: {gguf_path}")

        from llama_cpp import Llama

        self._model = Llama(model_path=str(model_path), n_ctx=n_ctx)
        self._create_completion = self._model.create_completion

    def generate(self, *, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        """Generate answer text and optional LLM abstain signal."""
        prompt = (
            "<SYSTEM>\n"
            f"{system_prompt}\n\n"
            "<USER>\n"
            f"{user_prompt}\n\n"
            "<FORMAT>\n"
            "반드시 JSON 객체 1개로만 답하세요."
            '키: answer(string), needs_abstain(boolean), reason(string).' 
            "JSON 외 텍스트를 포함하지 마세요.\n"
        )

        output = self._create_completion(
            prompt=prompt,
            temperature=self._temperature,
            top_p=self._top_p,
            max_tokens=self._max_tokens,
            repeat_penalty=self._repeat_penalty,
            stop=self._stop,
        )

        text = str(output["choices"][0]["text"]).strip()
        return _parse_llm_json(text)


def _parse_llm_json(text: str) -> dict[str, Any]:
    """Parse strict JSON contract; fallback to non-abstain plain answer."""
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text.replace("json", "", 1).strip()

    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return {
            "answer": text,
            "needs_abstain": False,
            "reason": "llm_output_not_json",
        }

    answer = str(payload.get("answer", "")).strip()
    needs_abstain = bool(payload.get("needs_abstain", False))
    reason = str(payload.get("reason", "")).strip()

    return {
        "answer": answer,
        "needs_abstain": needs_abstain,
        "reason": reason,
    }
