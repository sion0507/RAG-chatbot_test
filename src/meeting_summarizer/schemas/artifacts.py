"""Core artifact schemas for meeting summarization pipeline."""

from __future__ import annotations

from pydantic import BaseModel, Field, model_validator


class SegmentRecord(BaseModel):
    """Single source segment used for summarization."""

    segment_id: str = Field(min_length=1)
    speaker: str | None = None
    start_sec: float = Field(ge=0)
    end_sec: float = Field(gt=0)
    text: str = Field(min_length=1)

    @model_validator(mode="after")
    def _validate_time_range(self) -> "SegmentRecord":
        if self.end_sec <= self.start_sec:
            raise ValueError("segment end_sec must be greater than start_sec")
        return self


class SummaryCandidate(BaseModel):
    """Intermediate summary candidate output."""

    candidate_id: str = Field(min_length=1)
    segment_ids: list[str] = Field(min_length=1)
    content: str = Field(min_length=1)
    score: float | None = None


class SummaryCase(BaseModel):
    """Decision case containing selected candidates."""

    case_id: str = Field(min_length=1)
    candidate_ids: list[str] = Field(min_length=1)
    rationale: str | None = None


class MeetingSummaryArtifact(BaseModel):
    """Final fixed artifact for one meeting."""

    meeting_id: str = Field(min_length=1)
    language: str = Field(default="ko", min_length=2)
    title: str | None = None
    segments: list[SegmentRecord] = Field(default_factory=list)
    candidates: list[SummaryCandidate] = Field(default_factory=list)
    cases: list[SummaryCase] = Field(default_factory=list)
    final_summary: str = Field(min_length=1)
