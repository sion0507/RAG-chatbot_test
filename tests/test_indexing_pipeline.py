from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import faiss
import numpy as np

from src.embed.build_indexes import build_indexes


class IndexingPipelineTest(unittest.TestCase):
    def test_build_indexes_from_corpus(self) -> None:
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
                    "text": "워게임 모델 운용 지침 개요",
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
                    "chunk_tokens": 4,
                },
            ]
            corpus.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")

            def fake_embed(texts: list[str]) -> np.ndarray:
                return np.array([[float(i + 1), 1.0] for i, _ in enumerate(texts)], dtype=np.float32)

            summary = build_indexes(corpus, indexes, embed_texts=fake_embed)

            self.assertEqual(summary["chunks"], 2)
            self.assertTrue((indexes / "vector" / "faiss.index").exists())
            self.assertTrue((indexes / "bm25" / "bm25.pkl").exists())
            self.assertTrue((indexes / "chunks" / "chunks_store.jsonl").exists())

            index = faiss.read_index(str(indexes / "vector" / "faiss.index"))
            self.assertEqual(index.ntotal, 2)


if __name__ == "__main__":
    unittest.main()
