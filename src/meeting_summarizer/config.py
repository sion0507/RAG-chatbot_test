"""Config loader with explicit validation errors for meeting summarizer."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


class MeetingConfigError(ValueError):
    """Raised when summarizer config is missing or malformed."""


def _read_yaml(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.exists():
        raise MeetingConfigError(f"config file not found: {config_path}")

    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise MeetingConfigError(f"YAML root must be mapping: path={config_path}")
    return payload


def _require(config: dict[str, Any], keys: tuple[str, ...]) -> None:
    cursor: Any = config
    walked: list[str] = []
    for key in keys:
        walked.append(key)
        if not isinstance(cursor, dict) or key not in cursor:
            dotted = ".".join(walked)
            raise MeetingConfigError(f"missing required config: {dotted}")
        cursor = cursor[key]


def load_meeting_config(path: str | Path = "configs/meeting_summarizer.yaml") -> dict[str, Any]:
    """Load meeting summarizer config and validate required contract keys."""
    config = _read_yaml(path)
    required = [
        ("artifacts", "version"),
        ("artifacts", "output_dir"),
        ("id_rules", "meeting_prefix"),
        ("id_rules", "segment_prefix"),
        ("id_rules", "candidate_prefix"),
        ("id_rules", "case_prefix"),
    ]
    for keys in required:
        _require(config, keys)
    return config
