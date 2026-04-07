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
        n_gpu_layers: int = 0,
        main_gpu: int = 0,
        temperature: float,
        top_p: float,
        max_tokens: int,
        repeat_penalty: float,
        stop: list[str] | None,
        create_completion: Callable[..., dict[str, Any]] | None = None,
    ) -> None:
        if n_ctx <= 0:
            raise ValueError("n_ctx must be a positive integer.")
        if n_gpu_layers < -1:
            raise ValueError("n_gpu_layers must be -1(all) or non-negative integer.")
        if main_gpu < 0:
            raise ValueError("main_gpu must be a non-negative integer.")
        if max_tokens <= 0:
            raise ValueError("max_tokens must be a positive integer.")

        self._temperature = temperature
        self._top_p = top_p
        self._max_tokens = max_tokens
        self._repeat_penalty = repeat_penalty
        self._stop = _build_stop_sequences(stop)

        if create_completion is not None:
            self._create_completion = create_completion
            return

        model_path = Path(gguf_path)
        if not model_path.exists():
            raise FileNotFoundError(f"LLM gguf 파일을 찾을 수 없습니다: {gguf_path}")

        from llama_cpp import Llama

        self._model = Llama(
            model_path=str(model_path),
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            main_gpu=main_gpu,
        )
        self._create_completion = self._model.create_completion

    def generate(self, *, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        """Generate answer text and optional LLM abstain signal."""
        prompt = (
            f"{system_prompt}\n\n"
            f"{user_prompt}\n\n"
            "다음 스키마의 JSON 객체 1개만 출력하세요.\n"
            '{"answer": string, "needs_abstain": boolean, "reason": string}\n'
            "설명문, 마크다운, 코드블록, 추가 문장은 출력하지 마세요.\n"
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


def _build_stop_sequences(user_stop: list[str] | None) -> list[str]:
    """Return default + user stop sequences while preserving order."""
    defaults = ["</s>", "<SYSTEM>", "<USER>", "<FORMAT>", "<|im_end|>"]
    merged = [*(user_stop or []), *defaults]

    seen: set[str] = set()
    deduped: list[str] = []
    for item in merged:
        token = str(item).strip()
        if not token or token in seen:
            continue
        deduped.append(token)
        seen.add(token)
    return deduped


def _parse_llm_json(text: str) -> dict[str, Any]:
    """Parse JSON contract; extract first JSON object if extra text exists."""
    normalized = _strip_code_fence(text).strip()
    candidates = _extract_json_objects(normalized)

    if not candidates:
        return {
            "answer": "",
            "needs_abstain": True,
            "reason": "llm_output_not_json",
        }

    payload = candidates[0]
    answer = str(payload.get("answer", "")).strip()
    needs_abstain = bool(payload.get("needs_abstain", False))
    reason = str(payload.get("reason", "")).strip()

    return {
        "answer": answer,
        "needs_abstain": needs_abstain,
        "reason": reason,
    }


def _strip_code_fence(text: str) -> str:
    text = text.strip()
    if not text.startswith("```"):
        return text
    stripped = text.strip("`")
    if stripped.lower().startswith("json"):
        return stripped[4:].strip()
    return stripped.strip()


def _extract_json_objects(text: str) -> list[dict[str, Any]]:
    decoder = json.JSONDecoder()
    objects: list[dict[str, Any]] = []

    for idx, char in enumerate(text):
        if char != "{":
            continue
        try:
            parsed, _ = decoder.raw_decode(text[idx:])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            objects.append(parsed)

    return objects
