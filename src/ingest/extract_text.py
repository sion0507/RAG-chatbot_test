"""Document text extraction for ingest."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import fitz

from .normalize import normalize_text


@dataclass(frozen=True)
class PageText:
    """Extracted text payload for a page-like unit."""

    page_num: int
    text: str


class TextExtractionError(RuntimeError):
    """Raised when a document cannot be extracted."""



def extract_document_pages(path: str | Path) -> list[PageText]:
    """Extract page-wise text from PDF/MD/TXT document.

    Returns:
        List of normalized page blocks. For .md/.txt, a single page (1) is returned.
    """
    file_path = Path(path)
    suffix = file_path.suffix.lower()

    if suffix == ".pdf":
        return _extract_pdf(file_path)
    if suffix in {".md", ".txt"}:
        text = normalize_text(file_path.read_text(encoding="utf-8"))
        return [PageText(page_num=1, text=text)] if text else []

    raise TextExtractionError(f"Unsupported extension for extraction: {file_path.suffix}")


def _extract_pdf(path: Path) -> list[PageText]:
    pages: list[PageText] = []
    try:
        with fitz.open(path) as doc:
            for idx, page in enumerate(doc, start=1):
                text = normalize_text(page.get_text("text"))
                if text:
                    pages.append(PageText(page_num=idx, text=text))
    except Exception as exc:  # noqa: BLE001 - retain extraction root cause for operators
        raise TextExtractionError(f"Failed to extract PDF: {path} ({exc})") from exc

    return pages
