"""Chunk creation for section texts."""

from __future__ import annotations

from dataclasses import dataclass

from .parse_headings import Section


@dataclass(frozen=True)
class Chunk:
    """Minimal chunk payload expected by downstream indexing and QA."""

    doc_id: str
    section_path: str
    heading: str
    chunk_id: str
    chunk_index: int
    page_start: int
    page_end: int
    text: str
    chunk_tokens: int


def build_chunks(
    doc_id: str,
    sections: list[Section],
    max_chars: int = 1400,
    overlap_chars: int = 150,
    min_chars: int = 300,
) -> list[Chunk]:
    """Create stable chunks from parsed sections."""
    chunks: list[Chunk] = []

    for section in sections:
        if len(section.text) <= max_chars:
            part_texts = [section.text]
        else:
            part_texts = _split_with_overlap(section.text, max_chars=max_chars, overlap_chars=overlap_chars)

        for part_idx, part in enumerate(part_texts):
            if len(part) < min_chars and chunks and chunks[-1].section_path == section.section_path:
                prev = chunks[-1]
                merged = f"{prev.text}\n{part}".strip()
                chunks[-1] = Chunk(
                    doc_id=prev.doc_id,
                    section_path=prev.section_path,
                    heading=prev.heading,
                    chunk_id=prev.chunk_id,
                    chunk_index=prev.chunk_index,
                    page_start=prev.page_start,
                    page_end=section.page_end,
                    text=merged,
                    chunk_tokens=_estimate_tokens(merged),
                )
                continue

            chunk_index = len(chunks)
            chunk_id = f"{doc_id}:{chunk_index}"
            chunks.append(
                Chunk(
                    doc_id=doc_id,
                    section_path=section.section_path,
                    heading=section.heading,
                    chunk_id=chunk_id,
                    chunk_index=chunk_index,
                    page_start=section.page_start,
                    page_end=section.page_end,
                    text=part,
                    chunk_tokens=_estimate_tokens(part),
                )
            )

    return chunks


def _split_with_overlap(text: str, max_chars: int, overlap_chars: int) -> list[str]:
    chunks: list[str] = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = min(start + max_chars, text_len)
        candidate = text[start:end]
        if end < text_len:
            # try ending at nearby sentence boundary for readability
            for sep in ["\n", ". ", "다."]:
                split_pos = candidate.rfind(sep)
                if split_pos > max_chars * 0.5:
                    end = start + split_pos + len(sep)
                    candidate = text[start:end]
                    break

        chunks.append(candidate.strip())
        if end >= text_len:
            break
        start = max(0, end - overlap_chars)

    return [c for c in chunks if c]


def _estimate_tokens(text: str) -> int:
    # language-agnostic rough token estimate without external tokenizer dependency.
    return max(1, len(text.split()))
