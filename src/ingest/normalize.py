"""Text normalization helpers."""

from __future__ import annotations

import re

_WS_RE = re.compile(r"[ \t]+")
_MULTI_NEWLINE_RE = re.compile(r"\n{3,}")


def normalize_text(text: str) -> str:
    """Normalize whitespace while preserving paragraph boundaries."""
    # Normalize line endings first.
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    normalized = _WS_RE.sub(" ", normalized)
    normalized = _MULTI_NEWLINE_RE.sub("\n\n", normalized)
    return normalized.strip()
