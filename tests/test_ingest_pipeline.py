from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.ingest.build_corpus import build_corpus


class IngestPipelineTest(unittest.TestCase):
    def test_build_corpus_from_text_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = root / "raw"
            out = root / "processed" / "corpus.jsonl"
            raw.mkdir(parents=True, exist_ok=True)

            sample = raw / "manual.txt"
            sample.write_text(
                """1. 개요
이 문서는 워게임 모델 운용 지침입니다.
세부 절차를 확인합니다.

2. 절차
시뮬레이션 시작 전에 데이터 점검을 수행합니다.
""",
                encoding="utf-8",
            )

            docs, chunks = build_corpus(raw, out, max_chars=120, overlap_chars=20, min_chars=20)

            self.assertEqual(docs, 1)
            self.assertGreaterEqual(chunks, 1)
            lines = out.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(lines), chunks)

            payload = json.loads(lines[0])
            required = {
                "doc_id",
                "section_path",
                "heading",
                "chunk_id",
                "chunk_index",
                "page_start",
                "page_end",
                "text",
                "chunk_tokens",
            }
            self.assertTrue(required.issubset(payload.keys()))


if __name__ == "__main__":
    unittest.main()
