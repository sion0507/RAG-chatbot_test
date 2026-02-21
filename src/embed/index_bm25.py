"""BM25 index builder and serializer."""

from __future__ import annotations

import pickle
from pathlib import Path

from rank_bm25 import BM25Okapi


def tokenize(text: str) -> list[str]:
    """Simple whitespace tokenizer for Korean/English mixed text."""
    return [tok for tok in text.lower().split() if tok]


def build_bm25_index(texts: list[str]) -> tuple[BM25Okapi, list[list[str]]]:
    """Build BM25Okapi with tokenized corpus."""
    tokenized_corpus = [tokenize(text) for text in texts]
    if not tokenized_corpus:
        raise ValueError("Cannot build BM25 index from empty corpus.")
    return BM25Okapi(tokenized_corpus), tokenized_corpus


def save_bm25_index(index: BM25Okapi, tokenized_corpus: list[list[str]], output_path: str | Path) -> None:
    """Serialize BM25 artifacts with pickle for offline reuse."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "index": index,
        "tokenized_corpus": tokenized_corpus,
    }
    with out.open("wb") as fp:
        pickle.dump(payload, fp)
