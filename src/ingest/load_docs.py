"""Document discovery utilities for ingest pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable


SUPPORTED_EXTENSIONS = {".pdf", ".md", ".txt"}


def discover_documents(data_raw_dir: str | Path, extensions: Iterable[str] | None = None) -> list[Path]:
    """Return sorted list of ingestible files.

    Args:
        data_raw_dir: Root directory containing source documents.
        extensions: Optional iterable of allowed file extensions.

    Raises:
        FileNotFoundError: If ``data_raw_dir`` does not exist.

    Returns:
        Sorted ``Path`` list for stable corpus generation.
    """
    root = Path(data_raw_dir)
    if not root.exists():
        raise FileNotFoundError(f"Raw data directory not found: {root}")

    allowed = {ext.lower() for ext in (extensions or SUPPORTED_EXTENSIONS)}
    docs = [p for p in root.iterdir() if p.is_file() and p.suffix.lower() in allowed]
    return sorted(docs)
