"""Build vector/BM25 indexes from corpus.jsonl."""

from __future__ import annotations

import argparse
import logging
from collections.abc import Callable
from pathlib import Path

import numpy as np

from .embedder import SentenceTransformerEmbedder
from .index_bm25 import build_bm25_index, save_bm25_index
from .index_faiss import build_faiss_index, save_faiss_index
from .store import ChunkRecord, load_corpus, write_chunk_store, write_id_map

LOGGER = logging.getLogger(__name__)


def build_indexes(
    corpus_path: str | Path,
    output_root: str | Path = "indexes",
    *,
    build_vector: bool = True,
    build_bm25: bool = True,
    strict: bool = True,
    model_name: str = "dragonkue/multilingual-e5-small-ko-v2",
    model_local_dir: str | Path | None = "models/embedder",
    device: str = "cpu",
    batch_size: int = 16,
    normalize_embeddings: bool = True,
    embed_texts: Callable[[list[str]], np.ndarray] | None = None,
) -> dict[str, int]:
    """Build indexing artifacts.

    Returns:
        A summary dict including processed chunk count.
    """
    records = load_corpus(corpus_path, strict=strict)
    if not records:
        raise ValueError(f"No valid chunk records found in corpus: {corpus_path}")

    out_root = Path(output_root)
    chunk_store_path = out_root / "chunks" / "chunks_store.jsonl"
    write_chunk_store(records, chunk_store_path)

    texts = [rec.text for rec in records]

    if build_vector:
        vector_dir = out_root / "vector"
        encoder = embed_texts
        if encoder is None:
            encoder = SentenceTransformerEmbedder(
                model_name=model_name,
                local_dir=model_local_dir,
                device=device,
                batch_size=batch_size,
                normalize_embeddings=normalize_embeddings,
            ).encode

        vectors = encoder(texts)
        index = build_faiss_index(vectors)
        save_faiss_index(index, vector_dir / "faiss.index")
        write_id_map(records, vector_dir / "id_map.jsonl")
        LOGGER.info("벡터 인덱스 생성 완료: chunks=%d output=%s", len(records), vector_dir)

    if build_bm25:
        bm25_dir = out_root / "bm25"
        bm25, tokenized_corpus = build_bm25_index(texts)
        save_bm25_index(bm25, tokenized_corpus, bm25_dir / "bm25.pkl")
        write_id_map(records, bm25_dir / "id_map.jsonl")
        LOGGER.info("BM25 인덱스 생성 완료: chunks=%d output=%s", len(records), bm25_dir)

    summary = {
        "chunks": len(records),
        "vector": int(build_vector),
        "bm25": int(build_bm25),
    }
    LOGGER.info("인덱싱 완료: %s", summary)
    return summary


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build vector/BM25 indexes from corpus JSONL")
    parser.add_argument("--corpus", default="data/processed/corpus.jsonl")
    parser.add_argument("--output-root", default="indexes")
    parser.add_argument("--vector-only", action="store_true")
    parser.add_argument("--bm25-only", action="store_true")
    parser.add_argument("--non-strict", action="store_true")
    parser.add_argument("--model-name", default="dragonkue/multilingual-e5-small-ko-v2")
    parser.add_argument("--model-local-dir", default="models/embedder")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--no-normalize-embeddings", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser


def main() -> int:
    parser = _build_arg_parser()
    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, str(args.log_level).upper(), logging.INFO))

    if args.vector_only and args.bm25_only:
        parser.error("--vector-only and --bm25-only cannot be used together")

    build_vector = not args.bm25_only
    build_bm25 = not args.vector_only

    build_indexes(
        corpus_path=args.corpus,
        output_root=args.output_root,
        build_vector=build_vector,
        build_bm25=build_bm25,
        strict=not args.non_strict,
        model_name=args.model_name,
        model_local_dir=args.model_local_dir,
        device=args.device,
        batch_size=args.batch_size,
        normalize_embeddings=not args.no_normalize_embeddings,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
