from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.embed.build_indexes import build_indexes
from src.rag.retriever import HybridRetriever, _min_max_normalize


class RetrievalPipelineTest(unittest.TestCase):
    def test_min_max_normalize_returns_zero_when_all_scores_equal(self) -> None:
        normalized = _min_max_normalize([
            ("manual:0", 0.5),
            ("manual:1", 0.5),
            ("manual:2", 0.5),
        ])

        self.assertEqual(normalized["manual:0"], 0.0)
        self.assertEqual(normalized["manual:1"], 0.0)
        self.assertEqual(normalized["manual:2"], 0.0)

    def test_hybrid_retrieval_dedup_and_rerank_candidates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            corpus = root / "corpus.jsonl"
            indexes = root / "indexes"

            rows = [
                {
                    "doc_id": "manual",
                    "section_path": "1.개요",
                    "heading": "개요",
                    "chunk_id": "manual:0",
                    "chunk_index": 0,
                    "page_start": 1,
                    "page_end": 1,
                    "text": "워게임 모델 운용 개요",
                    "chunk_tokens": 4,
                },
                {
                    "doc_id": "manual",
                    "section_path": "2.절차",
                    "heading": "절차",
                    "chunk_id": "manual:1",
                    "chunk_index": 1,
                    "page_start": 2,
                    "page_end": 2,
                    "text": "시뮬레이션 시작 전 데이터 점검",
                    "chunk_tokens": 5,
                },
                {
                    "doc_id": "manual",
                    "section_path": "3.종료",
                    "heading": "종료",
                    "chunk_id": "manual:2",
                    "chunk_index": 2,
                    "page_start": 3,
                    "page_end": 3,
                    "text": "운용 종료 및 로그 확인",
                    "chunk_tokens": 4,
                },
            ]
            corpus.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")

            def fake_embed(texts: list[str]) -> np.ndarray:
                mapping = {
                    "워게임 모델 운용 개요": np.array([1.0, 0.0], dtype=np.float32),
                    "시뮬레이션 시작 전 데이터 점검": np.array([0.9, 0.1], dtype=np.float32),
                    "운용 종료 및 로그 확인": np.array([0.0, 1.0], dtype=np.float32),
                }
                return np.stack([mapping[text] for text in texts])

            build_indexes(corpus, indexes, embed_texts=fake_embed)

            retriever = HybridRetriever(
                chunk_store_path=indexes / "chunks" / "chunks_store.jsonl",
                vector_index_path=indexes / "vector" / "faiss.index",
                vector_id_map_path=indexes / "vector" / "id_map.jsonl",
                bm25_index_path=indexes / "bm25" / "bm25.pkl",
                bm25_id_map_path=indexes / "bm25" / "id_map.jsonl",
                embed_query=lambda _: np.array([1.0, 0.0], dtype=np.float32),
                use_vector=True,
                use_bm25=True,
                top_k_vector=3,
                top_k_bm25=3,
                hybrid_alpha=0.5,
                rerank_top_k=2,
            )

            result = retriever.retrieve("시뮬레이션 점검")

            self.assertGreaterEqual(len(result.candidates), 2)
            self.assertEqual(len(result.rerank_candidates), 2)

            ids = [cand.chunk.chunk_id for cand in result.candidates]
            self.assertEqual(len(ids), len(set(ids)))
            self.assertEqual(ids[0], "manual:1")

            self.assertGreaterEqual(result.candidates[0].final_score, result.candidates[1].final_score)

    def test_vector_only_retrieval(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            corpus = root / "corpus.jsonl"
            indexes = root / "indexes"

            rows = [
                {
                    "doc_id": "manual",
                    "section_path": "1.개요",
                    "heading": "개요",
                    "chunk_id": "manual:0",
                    "chunk_index": 0,
                    "page_start": 1,
                    "page_end": 1,
                    "text": "알파",
                    "chunk_tokens": 1,
                },
                {
                    "doc_id": "manual",
                    "section_path": "2.절차",
                    "heading": "절차",
                    "chunk_id": "manual:1",
                    "chunk_index": 1,
                    "page_start": 2,
                    "page_end": 2,
                    "text": "베타",
                    "chunk_tokens": 1,
                },
            ]
            corpus.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")

            def fake_embed(texts: list[str]) -> np.ndarray:
                mapping = {
                    "알파": np.array([1.0, 0.0], dtype=np.float32),
                    "베타": np.array([0.0, 1.0], dtype=np.float32),
                }
                return np.stack([mapping[text] for text in texts])

            build_indexes(corpus, indexes, embed_texts=fake_embed)

            retriever = HybridRetriever(
                chunk_store_path=indexes / "chunks" / "chunks_store.jsonl",
                vector_index_path=indexes / "vector" / "faiss.index",
                vector_id_map_path=indexes / "vector" / "id_map.jsonl",
                bm25_index_path=None,
                bm25_id_map_path=None,
                embed_query=lambda _: np.array([1.0, 0.0], dtype=np.float32),
                use_vector=True,
                use_bm25=False,
                top_k_vector=2,
                top_k_bm25=1,
                rerank_top_k=1,
            )

            result = retriever.retrieve("아무 질문")
            self.assertEqual(result.candidates[0].chunk.chunk_id, "manual:0")
            self.assertEqual(len(result.rerank_candidates), 1)


    def test_retrieve_with_reranker_uses_rerank_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            corpus = root / "corpus.jsonl"
            indexes = root / "indexes"

            rows = [
                {
                    "doc_id": "manual",
                    "section_path": "1.개요",
                    "heading": "개요",
                    "chunk_id": "manual:0",
                    "chunk_index": 0,
                    "page_start": 1,
                    "page_end": 1,
                    "text": "알파",
                    "chunk_tokens": 1,
                },
                {
                    "doc_id": "manual",
                    "section_path": "2.절차",
                    "heading": "절차",
                    "chunk_id": "manual:1",
                    "chunk_index": 1,
                    "page_start": 2,
                    "page_end": 2,
                    "text": "베타",
                    "chunk_tokens": 1,
                },
            ]
            corpus.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")

            def fake_embed(texts: list[str]) -> np.ndarray:
                mapping = {
                    "알파": np.array([1.0, 0.0], dtype=np.float32),
                    "베타": np.array([0.0, 1.0], dtype=np.float32),
                }
                return np.stack([mapping[text] for text in texts])

            build_indexes(corpus, indexes, embed_texts=fake_embed)

            retriever = HybridRetriever(
                chunk_store_path=indexes / "chunks" / "chunks_store.jsonl",
                vector_index_path=indexes / "vector" / "faiss.index",
                vector_id_map_path=indexes / "vector" / "id_map.jsonl",
                bm25_index_path=None,
                bm25_id_map_path=None,
                embed_query=lambda _: np.array([1.0, 0.0], dtype=np.float32),
                use_vector=True,
                use_bm25=False,
                top_k_vector=2,
                top_k_bm25=1,
                rerank_top_k=2,
            )

            class FakeReranker:
                def rerank(self, *, query: str, candidates: list, top_n: int) -> list:
                    _ = query
                    ordered = list(reversed(candidates))[:top_n]
                    return [
                        type("Obj", (), {
                            "chunk": item.chunk,
                            "retrieval_score": item.final_score,
                            "rerank_score": float(100 - idx),
                        })()
                        for idx, item in enumerate(ordered)
                    ]

            result = retriever.retrieve("아무 질문", reranker=FakeReranker(), final_top_n=1)
            self.assertTrue(result.used_reranker)
            self.assertEqual(result.final_candidates[0].chunk.chunk_id, "manual:1")

    def test_retrieve_fallback_when_reranker_errors(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            corpus = root / "corpus.jsonl"
            indexes = root / "indexes"

            rows = [
                {
                    "doc_id": "manual",
                    "section_path": "1.개요",
                    "heading": "개요",
                    "chunk_id": "manual:0",
                    "chunk_index": 0,
                    "page_start": 1,
                    "page_end": 1,
                    "text": "알파",
                    "chunk_tokens": 1,
                }
            ]
            corpus.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")

            def fake_embed(texts: list[str]) -> np.ndarray:
                return np.stack([np.array([1.0, 0.0], dtype=np.float32) for _ in texts])

            build_indexes(corpus, indexes, embed_texts=fake_embed)

            retriever = HybridRetriever(
                chunk_store_path=indexes / "chunks" / "chunks_store.jsonl",
                vector_index_path=indexes / "vector" / "faiss.index",
                vector_id_map_path=indexes / "vector" / "id_map.jsonl",
                bm25_index_path=None,
                bm25_id_map_path=None,
                embed_query=lambda _: np.array([1.0, 0.0], dtype=np.float32),
                use_vector=True,
                use_bm25=False,
                top_k_vector=1,
                top_k_bm25=1,
                rerank_top_k=1,
            )

            class BrokenReranker:
                def rerank(self, *, query: str, candidates: list, top_n: int) -> list:
                    raise RuntimeError("boom")

            result = retriever.retrieve("아무 질문", reranker=BrokenReranker(), final_top_n=1)
            self.assertFalse(result.used_reranker)
            self.assertEqual(result.final_candidates[0].chunk.chunk_id, "manual:0")


if __name__ == "__main__":
    unittest.main()
