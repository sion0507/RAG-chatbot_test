"""Application configuration loader for chat service wiring."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """Raised when required application config keys are missing."""


def _read_yaml(path: str | Path) -> dict[str, Any]:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise ConfigError(f"YAML root must be mapping: path={path}")
    return payload


def load_app_configs(
    *,
    rag_path: str | Path = "configs/rag.yaml",
    models_path: str | Path = "configs/models.yaml",
    prompts_path: str | Path = "configs/prompts.yaml",
) -> dict[str, Any]:
    """Load and validate app-level config payloads."""
    rag = _read_yaml(rag_path)
    models = _read_yaml(models_path)
    prompts = _read_yaml(prompts_path)

    required_rag_paths = [
        ("retrieval", "use_vector"),
        ("retrieval", "use_bm25"),
        ("retrieval", "top_k_vector"),
        ("retrieval", "top_k_bm25"),
        ("retrieval", "hybrid_alpha"),
        ("rerank", "rerank_top_k"),
        ("rerank", "final_top_n"),
        ("context", "max_context_chars"),
        ("context", "max_chunks_in_context"),
        ("grounding", "abstain_if_no_evidence"),
        ("grounding", "min_cited_chunks"),
    ]
    for root, key in required_rag_paths:
        if key not in rag.get(root, {}):
            raise ConfigError(f"missing config: rag.{root}.{key}")

    if "embedder" not in models or "reranker" not in models or "llm" not in models:
        raise ConfigError("missing config: models.embedder/reranker/llm")

    if "system" not in prompts or "user_template" not in prompts:
        raise ConfigError("missing config: prompts.system/user_template")

    return {
        "rag": rag,
        "models": models,
        "prompts": prompts,
    }
