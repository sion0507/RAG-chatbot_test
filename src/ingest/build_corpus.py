"""Build `data/processed/corpus.jsonl` from raw documents."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from .chunking import Chunk, build_chunks
from .extract_text import TextExtractionError, extract_document_pages
from .load_docs import discover_documents
from .parse_headings import split_into_sections

LOGGER = logging.getLogger(__name__)


def build_corpus(
    data_raw_dir: str | Path,
    output_path: str | Path,
    max_chars: int = 1400,
    overlap_chars: int = 150,
    min_chars: int = 300,
) -> tuple[int, int]:
    """Build corpus file.

    Returns:
        Tuple ``(document_count, chunk_count)``.
    """
    docs = discover_documents(data_raw_dir)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    total_chunks = 0
    with output.open("w", encoding="utf-8") as fp:
        for doc_path in docs:
            doc_id = doc_path.stem
            try:
                pages = extract_document_pages(doc_path)
            except TextExtractionError as exc:
                LOGGER.error("문서 추출 실패: file=%s error=%s", doc_path, exc)
                continue

            sections = split_into_sections(pages)
            chunks = build_chunks(
                doc_id=doc_id,
                sections=sections,
                max_chars=max_chars,
                overlap_chars=overlap_chars,
                min_chars=min_chars,
            )

            for chunk in chunks:
                fp.write(_chunk_to_json(chunk) + "\n")
            total_chunks += len(chunks)
            LOGGER.info("문서 처리 완료: file=%s chunks=%d", doc_path.name, len(chunks))

    LOGGER.info("코퍼스 생성 완료: docs=%d chunks=%d output=%s", len(docs), total_chunks, output)
    return len(docs), total_chunks


def _chunk_to_json(chunk: Chunk) -> str:
    payload = {
        "doc_id": chunk.doc_id,
        "section_path": chunk.section_path,
        "heading": chunk.heading,
        "chunk_id": chunk.chunk_id,
        "chunk_index": chunk.chunk_index,
        "page_start": chunk.page_start,
        "page_end": chunk.page_end,
        "text": chunk.text,
        "chunk_tokens": chunk.chunk_tokens,
    }
    return json.dumps(payload, ensure_ascii=False)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build corpus JSONL from raw docs")
    parser.add_argument("--data-raw-dir", default="data/raw")
    parser.add_argument("--output", default="data/processed/corpus.jsonl")
    parser.add_argument("--max-chars", type=int, default=1400)
    parser.add_argument("--overlap-chars", type=int, default=150)
    parser.add_argument("--min-chars", type=int, default=300)
    parser.add_argument("--log-level", default="INFO")
    return parser


def main() -> int:
    parser = _build_arg_parser()
    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, str(args.log_level).upper(), logging.INFO))

    _, chunk_count = build_corpus(
        data_raw_dir=args.data_raw_dir,
        output_path=args.output,
        max_chars=args.max_chars,
        overlap_chars=args.overlap_chars,
        min_chars=args.min_chars,
    )
    return 0 if chunk_count >= 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
