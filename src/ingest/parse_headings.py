"""Heading-aware section parser."""

from __future__ import annotations

from dataclasses import dataclass
import re

from .extract_text import PageText


_MD_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
_NUMBER_HEADING_RE = re.compile(r"^(\d+(?:\.\d+)*)[\.)]?\s*(.+)$")
_SIMPLE_NUMBERED_ITEM_RE = re.compile(r"^(\d+)[\.)]\s+(.+)$")


@dataclass(frozen=True)
class Section:
    """Logical section with inherited heading path."""

    section_path: str
    heading: str
    page_start: int
    page_end: int
    text: str


def split_into_sections(pages: list[PageText]) -> list[Section]:
    """Split page text into heading-aware sections.

    If no heading markers are detected, one section per page is created.
    """
    sections: list[Section] = []
    stack: list[str] = []

    for page in pages:
        current_heading = "본문"
        buffer: list[str] = []
        lines = [line.strip() for line in page.text.splitlines() if line.strip()]

        def flush_buffer() -> None:
            content = "\n".join(buffer).strip()
            if not content:
                return
            path_parts = stack or [current_heading]
            sections.append(
                Section(
                    section_path=" > ".join(path_parts),
                    heading=path_parts[-1],
                    page_start=page.page_num,
                    page_end=page.page_num,
                    text=content,
                )
            )

        for line_idx, line in enumerate(lines):
            md_match = _MD_HEADING_RE.match(line)
            if md_match:
                flush_buffer()
                depth = len(md_match.group(1))
                title = md_match.group(2).strip()
                stack = stack[: depth - 1]
                stack.append(title)
                current_heading = title
                buffer = []
                continue

            num_match = _NUMBER_HEADING_RE.match(line)
            if num_match and len(line) < 120:
                number = num_match.group(1)
                title = num_match.group(2).strip()

                is_simple_numbered_item = "." not in number
                if is_simple_numbered_item and _looks_like_procedural_sequence(lines, line_idx):
                    buffer.append(line)
                    continue

                flush_buffer()
                depth = number.count(".") + 1
                full_title = f"{number} {title}".strip()
                stack = stack[: depth - 1]
                stack.append(full_title)
                current_heading = full_title
                buffer = []
                continue

            buffer.append(line)

        flush_buffer()

    return sections


def _looks_like_procedural_sequence(lines: list[str], current_idx: int) -> bool:
    """Heuristic guard to avoid misclassifying `1. 2. 3.` steps as new sections."""
    current = _extract_simple_index(lines[current_idx])
    if current is None:
        return False

    nearby = lines[max(0, current_idx - 2) : min(len(lines), current_idx + 3)]
    nearby_indices = [idx for line in nearby if (idx := _extract_simple_index(line)) is not None]
    if not nearby_indices:
        return False

    # classify as a procedural list when adjacent indices are visible in near lines.
    return (current + 1 in nearby_indices) or (current - 1 in nearby_indices)


def _extract_simple_index(line: str) -> int | None:
    match = _SIMPLE_NUMBERED_ITEM_RE.match(line)
    if not match:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None
