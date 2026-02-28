"""Hybrid retrieval over FAISS and BM25 artifacts."""

from __future__ import annotations

import json
import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import faiss
import numpy as np

from src.embed.index_bm25 import tokenize
from src.embed.store import ChunkRecord, load_corpus

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class RetrievalCandidate:
    """Candidate chunk with source and fused scores."""

    chunk: ChunkRecord
    vector_score: float
    bm25_score: float
    final_score: float


@dataclass(frozen=True)
class RetrievalResult:
    """Retrieval output used as reranker input."""

    query: str
    candidates: list[RetrievalCandidate]
    rerank_candidates: list[RetrievalCandidate]


class HybridRetriever:
    """Load retrieval artifacts and serve hybrid candidates."""

    def __init__(
        self,
        *,
        chunk_store_path: str | Path,
        vector_index_path: str | Path | None,
        vector_id_map_path: str | Path | None,
        bm25_index_path: str | Path | None,
        bm25_id_map_path: str | Path | None,
        embed_query: Callable[[str], np.ndarray],
        use_vector: bool = True,
        use_bm25: bool = True,
        top_k_vector: int = 20,
        top_k_bm25: int = 20,
        hybrid_alpha: float = 0.5,
        rerank_top_k: int = 8,
    ) -> None:
        if not use_vector and not use_bm25:
            raise ValueError("At least one retrieval backend must be enabled.")
        if not 0.0 <= hybrid_alpha <= 1.0:
            raise ValueError("hybrid_alpha must be in [0, 1].")
        if top_k_vector <= 0 or top_k_bm25 <= 0:
            raise ValueError("top_k_vector/top_k_bm25 must be positive integers.")
        if rerank_top_k <= 0:
            raise ValueError("rerank_top_k must be a positive integer.")

        self._use_vector = use_vector
        self._use_bm25 = use_bm25
        self._top_k_vector = top_k_vector
        self._top_k_bm25 = top_k_bm25
        self._hybrid_alpha = hybrid_alpha
        self._rerank_top_k = rerank_top_k
        self._embed_query = embed_query

        self._chunks_by_id = _load_chunks_by_id(chunk_store_path)

        self._vector_index = None
        self._vector_id_map: list[str] = []
        if use_vector:
            if vector_index_path is None or vector_id_map_path is None:
                raise ValueError("Vector index/id_map paths are required when use_vector=True.")
            self._vector_index = faiss.read_index(str(vector_index_path))
            self._vector_id_map = _load_index_id_map(vector_id_map_path)

        self._bm25_index = None
        self._bm25_tokenized_corpus: list[list[str]] = []
        self._bm25_id_map: list[str] = []
        if use_bm25:
            if bm25_index_path is None or bm25_id_map_path is None:
                raise ValueError("BM25 index/id_map paths are required when use_bm25=True.")
            with Path(bm25_index_path).open("rb") as fp:
                payload = pickle.load(fp)
            self._bm25_index = payload["index"]
            self._bm25_tokenized_corpus = payload["tokenized_corpus"]
            self._bm25_id_map = _load_index_id_map(bm25_id_map_path)

    def retrieve(self, query: str) -> RetrievalResult:
        """Search by query and return sorted candidates and rerank subset."""
        merged: dict[str, dict[str, float]] = {}

        if self._use_vector:
            vector_scores = self._vector_search(query)
            for chunk_id, score in vector_scores.items():
                merged.setdefault(chunk_id, {"vector": 0.0, "bm25": 0.0})["vector"] = score

        if self._use_bm25:
            bm25_scores = self._bm25_search(query)
            for chunk_id, score in bm25_scores.items():
                merged.setdefault(chunk_id, {"vector": 0.0, "bm25": 0.0})["bm25"] = score

        candidates: list[RetrievalCandidate] = []
        for chunk_id, scores in merged.items():
            chunk = self._chunks_by_id.get(chunk_id)
            if chunk is None:
                LOGGER.warning("chunk_id 누락으로 후보 제외: chunk_id=%s", chunk_id)
                continue

            final_score = self._hybrid_alpha * scores["vector"] + (1.0 - self._hybrid_alpha) * scores["bm25"]
            candidates.append(
                RetrievalCandidate(
                    chunk=chunk,
                    vector_score=scores["vector"],
                    bm25_score=scores["bm25"],
                    final_score=final_score,
                )
            )

        candidates.sort(key=lambda item: item.final_score, reverse=True)
        rerank_candidates = candidates[: self._rerank_top_k]

        return RetrievalResult(query=query, candidates=candidates, rerank_candidates=rerank_candidates)

    def _vector_search(self, query: str) -> dict[str, float]:
        query_vector = self._embed_query(query)
        matrix = np.asarray(query_vector, dtype=np.float32)
        if matrix.ndim == 1:
            matrix = matrix.reshape(1, -1)

        if matrix.ndim != 2 or matrix.shape[0] != 1:
            raise ValueError("embed_query must return shape (d,) or (1, d).")

        scores, indices = self._vector_index.search(matrix, self._top_k_vector)
        pairs = []
        for idx, score in zip(indices[0], scores[0], strict=True):
            if idx < 0:
                continue
            chunk_id = _safe_get_chunk_id(self._vector_id_map, int(idx), source="vector")
            pairs.append((chunk_id, float(score)))
        return _min_max_normalize(pairs)

    def _bm25_search(self, query: str) -> dict[str, float]:
        query_tokens = tokenize(query)
        if not query_tokens:
            return {}

        raw_scores = self._bm25_index.get_scores(query_tokens)
        if raw_scores.size == 0:
            return {}

        top_k = min(self._top_k_bm25, raw_scores.shape[0])
        top_indices = np.argsort(raw_scores)[::-1][:top_k]

        pairs = []
        for idx in top_indices:
            chunk_id = _safe_get_chunk_id(self._bm25_id_map, int(idx), source="bm25")
            pairs.append((chunk_id, float(raw_scores[idx])))
        return _min_max_normalize(pairs)


def _min_max_normalize(pairs: list[tuple[str, float]]) -> dict[str, float]:
    if not pairs:
        return {}

    values = [score for _, score in pairs]
    lo = min(values)
    hi = max(values)

    if np.isclose(hi, lo):
        return {chunk_id: 0.0 for chunk_id, _ in pairs}

    return {chunk_id: (score - lo) / (hi - lo) for chunk_id, score in pairs}


def _safe_get_chunk_id(id_map: list[str], index_id: int, source: str) -> str:
    try:
        return id_map[index_id]
    except IndexError as exc:
        raise ValueError(f"{source} id_map 누락: index_id={index_id}") from exc


def _load_index_id_map(path: str | Path) -> list[str]:
    ids: list[str] = []
    with Path(path).open("r", encoding="utf-8") as fp:
        for line in fp:
            payload = json.loads(line)
            ids.append(str(payload["chunk_id"]))
    return ids


def _load_chunks_by_id(path: str | Path) -> dict[str, ChunkRecord]:
    chunks = load_corpus(path, strict=True)
    out: dict[str, ChunkRecord] = {}
    for chunk in chunks:
        out[chunk.chunk_id] = chunk
    return out
