"""FAISS vector index builder."""

from __future__ import annotations

from pathlib import Path

import faiss
import numpy as np


def build_faiss_index(vectors: np.ndarray) -> faiss.IndexFlatIP:
    """Build inner-product index from normalized vectors."""
    if vectors.ndim != 2 or vectors.shape[0] == 0:
        raise ValueError("Vector matrix must be 2D with at least one row.")

    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    return index


def save_faiss_index(index: faiss.IndexFlatIP, output_path: str | Path) -> None:
    """Persist FAISS index to disk."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(out))
