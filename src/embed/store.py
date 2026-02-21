"""Corpus loading and index metadata store utilities."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

REQUIRED_CHUNK_FIELDS = {
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


@dataclass(frozen=True)
class ChunkRecord:
    """Canonical chunk payload used by indexing and retrieval."""

    doc_id: str
    section_path: str
    heading: str
    chunk_id: str
    chunk_index: int
    page_start: int
    page_end: int
    text: str
    chunk_tokens: int


class CorpusValidationError(ValueError):
    """Raised when a corpus row is missing required fields."""


def load_corpus(path: str | Path, strict: bool = True) -> list[ChunkRecord]:
    """Load corpus JSONL and validate required fields.

    Args:
        path: Input corpus JSONL path.
        strict: When true, raises on invalid rows. Otherwise invalid rows are skipped.
    """
    records: list[ChunkRecord] = []
    file_path = Path(path)

    with file_path.open("r", encoding="utf-8") as fp:
        for line_no, line in enumerate(fp, start=1):
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            missing = REQUIRED_CHUNK_FIELDS - payload.keys()
            if missing:
                if strict:
                    raise CorpusValidationError(
                        f"Invalid corpus row at line {line_no}: missing fields={sorted(missing)}"
                    )
                continue

            records.append(
                ChunkRecord(
                    doc_id=str(payload["doc_id"]),
                    section_path=str(payload["section_path"]),
                    heading=str(payload["heading"]),
                    chunk_id=str(payload["chunk_id"]),
                    chunk_index=int(payload["chunk_index"]),
                    page_start=int(payload["page_start"]),
                    page_end=int(payload["page_end"]),
                    text=str(payload["text"]),
                    chunk_tokens=int(payload["chunk_tokens"]),
                )
            )

    return records


def write_chunk_store(records: Iterable[ChunkRecord], output_path: str | Path) -> None:
    """Write canonical chunk store JSONL."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    with out.open("w", encoding="utf-8") as fp:
        for rec in records:
            fp.write(json.dumps(asdict(rec), ensure_ascii=False) + "\n")


def write_id_map(records: Iterable[ChunkRecord], output_path: str | Path) -> None:
    """Write ordered index ID mapping for retrieval."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    with out.open("w", encoding="utf-8") as fp:
        for idx, rec in enumerate(records):
            payload = {
                "index_id": idx,
                "chunk_id": rec.chunk_id,
                "doc_id": rec.doc_id,
                "section_path": rec.section_path,
                "heading": rec.heading,
                "chunk_index": rec.chunk_index,
                "page_start": rec.page_start,
                "page_end": rec.page_end,
            }
            fp.write(json.dumps(payload, ensure_ascii=False) + "\n")
