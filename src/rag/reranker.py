"""Cross-encoder reranker for retrieval candidates."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable

import numpy as np

from src.rag.retriever import RetrievalCandidate

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class RerankCandidate:
    """Reranked candidate with retrieval and rerank scores."""

    chunk: object
    retrieval_score: float
    rerank_score: float


class CrossEncoderReranker:
    """CPU-first reranker based on sentence-transformers CrossEncoder."""

    def __init__(
        self,
        *,
        model_name_or_path: str,
        device: str = "cpu",
        batch_size: int = 8,
        max_length: int = 512,
        score_pairs: Callable[[list[tuple[str, str]]], np.ndarray] | None = None,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be a positive integer.")
        if max_length <= 0:
            raise ValueError("max_length must be a positive integer.")

        self._batch_size = batch_size

        if score_pairs is not None:
            self._score_pairs = score_pairs
            return

        try:
            from sentence_transformers import CrossEncoder
        except Exception as exc:  # pragma: no cover - depends on runtime env
            raise RuntimeError("sentence-transformers CrossEncoder를 불러올 수 없습니다.") from exc

        LOGGER.info(
            "리랭커 모델 로딩: model=%s device=%s batch_size=%s max_length=%s",
            model_name_or_path,
            device,
            batch_size,
            max_length,
        )
        self._model = CrossEncoder(model_name_or_path, device=device, max_length=max_length)

        def _predict(pairs: list[tuple[str, str]]) -> np.ndarray:
            scores = self._model.predict(
                pairs,
                batch_size=self._batch_size,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
            return np.asarray(scores, dtype=np.float32)

        self._score_pairs = _predict

    def rerank(
        self,
        *,
        query: str,
        candidates: list[RetrievalCandidate],
        top_n: int,
    ) -> list[RerankCandidate]:
        """Rerank candidates and return top_n by rerank score descending."""
        if top_n <= 0:
            raise ValueError("top_n must be a positive integer.")
        if not candidates:
            return []

        pairs = [(query, candidate.chunk.text) for candidate in candidates]
        scores = self._score_pairs(pairs)

        if len(scores) != len(candidates):
            raise ValueError("리랭커 점수 개수가 후보 개수와 일치하지 않습니다.")

        reranked = [
            RerankCandidate(
                chunk=candidate.chunk,
                retrieval_score=candidate.final_score,
                rerank_score=float(score),
            )
            for candidate, score in zip(candidates, scores, strict=True)
        ]
        reranked.sort(key=lambda item: item.rerank_score, reverse=True)
        return reranked[:top_n]
