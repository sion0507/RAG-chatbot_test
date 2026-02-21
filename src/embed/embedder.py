"""Embedding model wrapper used by vector indexing."""

from __future__ import annotations

from pathlib import Path

import numpy as np


class SentenceTransformerEmbedder:
    """Thin wrapper around sentence-transformers for deterministic indexing calls."""

    def __init__(
        self,
        model_name: str,
        device: str = "cpu",
        batch_size: int = 16,
        normalize_embeddings: bool = True,
        local_dir: str | Path | None = None,
    ) -> None:
        from sentence_transformers import SentenceTransformer

        model_source = str(local_dir) if local_dir and Path(local_dir).exists() else model_name
        self._model = SentenceTransformer(model_source, device=device)
        self._batch_size = batch_size
        self._normalize_embeddings = normalize_embeddings

    def encode(self, texts: list[str]) -> np.ndarray:
        """Encode text list into float32 embedding matrix."""
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)

        vectors = self._model.encode(
            texts,
            batch_size=self._batch_size,
            normalize_embeddings=self._normalize_embeddings,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float32)
