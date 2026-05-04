"""Deterministic ID generation rules for meeting summarizer entities."""

from __future__ import annotations

import hashlib
import re

_SLUG_RE = re.compile(r"[^a-z0-9]+")


def _slug(value: str) -> str:
    normalized = value.strip().lower()
    normalized = _SLUG_RE.sub("-", normalized).strip("-")
    return normalized or "unknown"


def build_meeting_id(*, title: str, date_iso: str, prefix: str = "mtg") -> str:
    return f"{prefix}_{date_iso}_{_slug(title)}"


def build_segment_id(*, meeting_id: str, index: int, prefix: str = "seg") -> str:
    if index < 0:
        raise ValueError("segment index must be >= 0")
    return f"{prefix}_{meeting_id}_{index:04d}"


def build_candidate_id(*, meeting_id: str, segment_ids: list[str], prefix: str = "cand") -> str:
    if not segment_ids:
        raise ValueError("segment_ids must not be empty")
    digest = hashlib.sha1("|".join(segment_ids).encode("utf-8")).hexdigest()[:10]
    return f"{prefix}_{meeting_id}_{digest}"


def build_case_id(*, meeting_id: str, candidate_ids: list[str], prefix: str = "case") -> str:
    if not candidate_ids:
        raise ValueError("candidate_ids must not be empty")
    digest = hashlib.sha1("|".join(sorted(candidate_ids)).encode("utf-8")).hexdigest()[:10]
    return f"{prefix}_{meeting_id}_{digest}"
