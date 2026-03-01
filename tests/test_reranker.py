from __future__ import annotations

import unittest

import numpy as np

from src.embed.store import ChunkRecord
from src.rag.reranker import CrossEncoderReranker
from src.rag.retriever import RetrievalCandidate


class RerankerTest(unittest.TestCase):
    def _candidate(self, chunk_id: str, text: str, retrieval_score: float) -> RetrievalCandidate:
        return RetrievalCandidate(
            chunk=ChunkRecord(
                doc_id="manual",
                section_path="sec",
                heading="head",
                chunk_id=chunk_id,
                chunk_index=0,
                page_start=1,
                page_end=1,
                text=text,
                chunk_tokens=1,
            ),
            vector_score=retrieval_score,
            bm25_score=0.0,
            final_score=retrieval_score,
        )

    def test_rerank_orders_by_descending_score(self) -> None:
        candidates = [
            self._candidate("manual:0", "알파", 0.9),
            self._candidate("manual:1", "베타", 0.8),
            self._candidate("manual:2", "감마", 0.7),
        ]

        reranker = CrossEncoderReranker(
            model_name_or_path="local-model",
            score_pairs=lambda pairs: np.array([0.1, 0.9, 0.4], dtype=np.float32),
        )

        result = reranker.rerank(query="질문", candidates=candidates, top_n=3)
        self.assertEqual([item.chunk.chunk_id for item in result], ["manual:1", "manual:2", "manual:0"])

    def test_rerank_applies_top_n(self) -> None:
        candidates = [
            self._candidate("manual:0", "알파", 0.9),
            self._candidate("manual:1", "베타", 0.8),
        ]
        reranker = CrossEncoderReranker(
            model_name_or_path="local-model",
            score_pairs=lambda pairs: np.array([0.4, 0.6], dtype=np.float32),
        )

        result = reranker.rerank(query="질문", candidates=candidates, top_n=1)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].chunk.chunk_id, "manual:1")

    def test_rerank_returns_empty_for_empty_candidates(self) -> None:
        reranker = CrossEncoderReranker(
            model_name_or_path="local-model",
            score_pairs=lambda pairs: np.array([], dtype=np.float32),
        )
        result = reranker.rerank(query="질문", candidates=[], top_n=3)
        self.assertEqual(result, [])


if __name__ == "__main__":
    unittest.main()
