"""Context building utilities for LLM prompting."""

from __future__ import annotations

from dataclasses import dataclass

from src.rag.retriever import FinalCandidate


@dataclass(frozen=True)
class ContextBuildResult:
    """LLM-ready evidence context and selected candidates."""

    context: str
    selected: list[FinalCandidate]


class ContextBuilder:
    """Construct bounded context text from final retrieval candidates."""

    def __init__(
        self,
        *,
        max_context_chars: int,
        max_chunks_in_context: int,
        include_chunk_header: bool,
    ) -> None:
        if max_context_chars <= 0:
            raise ValueError("max_context_chars must be a positive integer.")
        if max_chunks_in_context <= 0:
            raise ValueError("max_chunks_in_context must be a positive integer.")

        self._max_context_chars = max_context_chars
        self._max_chunks_in_context = max_chunks_in_context
        self._include_chunk_header = include_chunk_header

    def build(self, candidates: list[FinalCandidate]) -> ContextBuildResult:
        """Build context block from ranked candidates while enforcing char budget."""
        selected: list[FinalCandidate] = []
        chunks: list[str] = []
        used = 0

        for rank, candidate in enumerate(candidates, start=1):
            if len(selected) >= self._max_chunks_in_context:
                break

            block = self._render_chunk(rank=rank, candidate=candidate)
            if used + len(block) > self._max_context_chars:
                remaining = self._max_context_chars - used
                if remaining <= 0:
                    break
                block = block[:remaining]

            chunks.append(block)
            selected.append(candidate)
            used += len(block)

            if used >= self._max_context_chars:
                break

        return ContextBuildResult(context="\n\n".join(chunks).strip(), selected=selected)

    def _render_chunk(self, *, rank: int, candidate: FinalCandidate) -> str:
        chunk = candidate.chunk
        if not self._include_chunk_header:
            return chunk.text

        return (
            f"[CHUNK {rank}]\n"
            f"doc_id: {chunk.doc_id}\n"
            f"section_path: {chunk.section_path}\n"
            f"page: {chunk.page_start}-{chunk.page_end}\n"
            f"chunk_id: {chunk.chunk_id}\n"
            f"text: {chunk.text}"
        )
