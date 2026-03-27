"""Dependency builders for FastAPI routes."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from src.app.config import load_app_configs
from src.app.services.chat_service import ChatService, GroundingPolicy
from src.app.services.context_builder import ContextBuilder
from src.embed.embedder import SentenceTransformerEmbedder
from src.llm.llama_cpp_client import LlamaCppClient
from src.rag.reranker import CrossEncoderReranker
from src.rag.retriever import HybridRetriever


@lru_cache(maxsize=1)
def get_chat_service() -> ChatService:
    """Create singleton chat service for app runtime."""
    configs = load_app_configs()
    rag = configs["rag"]
    models = configs["models"]
    prompts = configs["prompts"]

    embed_cfg = models["embedder"]
    rerank_cfg = models["reranker"]
    llm_cfg = models["llm"]

    embedder = SentenceTransformerEmbedder(
        model_name=embed_cfg["name"],
        local_dir=embed_cfg.get("local_dir"),
        device=embed_cfg.get("device", "cpu"),
        batch_size=int(embed_cfg.get("batch_size", 16)),
        normalize_embeddings=bool(embed_cfg.get("normalize_embeddings", True)),
    )

    reranker_model_path = str(rerank_cfg.get("local_dir"))
    if not Path(reranker_model_path).exists():
        reranker_model_path = str(rerank_cfg["name"])

    retriever = HybridRetriever(
        chunk_store_path="indexes/chunks/chunks_store.jsonl",
        vector_index_path="indexes/vector/faiss.index" if rag["retrieval"]["use_vector"] else None,
        vector_id_map_path="indexes/vector/id_map.jsonl" if rag["retrieval"]["use_vector"] else None,
        bm25_index_path="indexes/bm25/bm25.pkl" if rag["retrieval"]["use_bm25"] else None,
        bm25_id_map_path="indexes/bm25/id_map.jsonl" if rag["retrieval"]["use_bm25"] else None,
        embed_query=lambda query: embedder.encode([query])[0],
        use_vector=bool(rag["retrieval"]["use_vector"]),
        use_bm25=bool(rag["retrieval"]["use_bm25"]),
        top_k_vector=int(rag["retrieval"]["top_k_vector"]),
        top_k_bm25=int(rag["retrieval"]["top_k_bm25"]),
        hybrid_alpha=float(rag["retrieval"]["hybrid_alpha"]),
        rerank_top_k=int(rag["rerank"]["rerank_top_k"]),
    )

    reranker = CrossEncoderReranker(
        model_name_or_path=reranker_model_path,
        device=rerank_cfg.get("device", "cpu"),
        batch_size=int(rerank_cfg.get("batch_size", 8)),
        max_length=int(rerank_cfg.get("max_length", 512)),
    )

    llm_client = LlamaCppClient(
        gguf_path=llm_cfg["gguf_path"],
        n_ctx=int(llm_cfg.get("n_ctx", 4096)),
        temperature=float(llm_cfg.get("temperature", 0.2)),
        top_p=float(llm_cfg.get("top_p", 0.9)),
        max_tokens=int(llm_cfg.get("max_tokens", 512)),
        repeat_penalty=float(llm_cfg.get("repeat_penalty", 1.1)),
        stop=list(llm_cfg.get("stop", [])),
    )

    context_builder = ContextBuilder(
        max_context_chars=int(rag["context"]["max_context_chars"]),
        max_chunks_in_context=int(rag["context"]["max_chunks_in_context"]),
        include_chunk_header=bool(rag["context"].get("include_chunk_header", True)),
    )

    policy = GroundingPolicy(
        abstain_if_no_evidence=bool(rag["grounding"].get("abstain_if_no_evidence", True)),
        min_cited_chunks=int(rag["grounding"].get("min_cited_chunks", 1)),
        llm_abstain_enabled=bool(rag["grounding"].get("llm_abstain_enabled", True)),
    )

    return ChatService(
        retriever=retriever,
        reranker=reranker,
        llm_client=llm_client,
        context_builder=context_builder,
        system_prompt=str(prompts["system"]),
        user_template=str(prompts["user_template"]),
        context_intro=str(prompts.get("context_intro", "")),
        policy=policy,
        final_top_n=int(rag["rerank"]["final_top_n"]),
    )
