"""BM25 index builder and serializer."""

from __future__ import annotations

import pickle
from pathlib import Path

from kiwipiepy import Kiwi
from rank_bm25 import BM25Okapi

# 전역 1회 초기화(매 tokenize 호출마다 만들면 느림)
_KIWI = Kiwi()

# BM25용: 명사 + (동사/형용사/보조용언) + 영문/숫자
_KEEP_TAGS = {
    "NNG",
    "NNP",
    "NNB",
    "NR",
    "NP",  # 명사류
    "VV",
    "VA",
    "VX",  # 동사/형용사/보조용언
    "SL",
    "SN",  # 영문/숫자
}

# 토큰이 비어버린 문서를 위한 더미 토큰
_DUMMY_TOKEN = "__empty__"


def tokenize(text: str) -> list[str]:
    """Kiwi tokenizer for Korean/English mixed text (BM25)."""
    text = text.lower()
    toks = _KIWI.tokenize(text, normalize_coda=True)
    return [t.form for t in toks if t.tag in _KEEP_TAGS and t.form]


def build_bm25_index(texts: list[str]) -> tuple[BM25Okapi, list[list[str]]]:
    """Build BM25Okapi with tokenized corpus."""
    if not texts:
        raise ValueError("Cannot build BM25 index from empty corpus.")

    tokenized_corpus: list[list[str]] = []
    for text in texts:
        toks = tokenize(text)
        if not toks:
            toks = [_DUMMY_TOKEN]
        tokenized_corpus.append(toks)

    return BM25Okapi(tokenized_corpus), tokenized_corpus


def save_bm25_index(
    index: BM25Okapi,
    tokenized_corpus: list[list[str]],
    output_path: str | Path,
) -> None:
    """Serialize BM25 artifacts with pickle for offline reuse."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "index": index,
        "tokenized_corpus": tokenized_corpus,
    }
    with out.open("wb") as fp:
        pickle.dump(payload, fp)
