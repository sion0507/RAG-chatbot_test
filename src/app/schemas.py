"""HTTP request/response schemas for chat application."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """Client request payload for chat endpoint."""

    question: str = Field(min_length=1, description="User question in natural language")
    top_n: int | None = Field(default=None, ge=1, le=10)


class Citation(BaseModel):
    """Structured citation metadata."""

    doc_id: str
    section_path: str
    page_start: int
    page_end: int
    chunk_id: str


class EvidenceItem(BaseModel):
    """Supporting evidence with rank scores."""

    doc_id: str
    section_path: str
    page_start: int
    page_end: int
    chunk_id: str
    retrieval_score: float | None
    rerank_score: float | None
    text: str


class ChatResponse(BaseModel):
    """Structured response including answer and evidence."""

    answer: str
    citations: list[Citation]
    evidence: list[EvidenceItem]
    abstained: bool
    abstain_source: str | None = None
    abstain_reason: str | None = None


class HealthResponse(BaseModel):
    """Health check response payload."""

    status: str
    components: dict[str, str]
