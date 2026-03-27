"""Chat orchestration service for retrieval-grounded answer generation."""

from __future__ import annotations

from dataclasses import dataclass

from src.app.schemas import ChatResponse, Citation, EvidenceItem
from src.app.services.context_builder import ContextBuilder
from src.rag.retriever import HybridRetriever


@dataclass(frozen=True)
class GroundingPolicy:
    """Grounding and abstention policy configuration."""

    abstain_if_no_evidence: bool
    min_cited_chunks: int
    llm_abstain_enabled: bool = True


class ChatService:
    """Coordinate retriever, reranker, context builder, and LLM."""

    def __init__(
        self,
        *,
        retriever: HybridRetriever,
        reranker: object,
        llm_client: object,
        context_builder: ContextBuilder,
        system_prompt: str,
        user_template: str,
        context_intro: str,
        policy: GroundingPolicy,
        final_top_n: int,
    ) -> None:
        if final_top_n <= 0:
            raise ValueError("final_top_n must be a positive integer.")

        self._retriever = retriever
        self._reranker = reranker
        self._llm_client = llm_client
        self._context_builder = context_builder
        self._system_prompt = system_prompt
        self._user_template = user_template
        self._context_intro = context_intro
        self._policy = policy
        self._final_top_n = final_top_n

    def ask(self, question: str, top_n: int | None = None) -> ChatResponse:
        """Return grounded answer with citations and evidence payload."""
        query = question.strip()
        if not query:
            raise ValueError("question must not be empty.")

        effective_top_n = top_n if top_n is not None else self._final_top_n
        result = self._retriever.retrieve(query, reranker=self._reranker, final_top_n=effective_top_n)

        build = self._context_builder.build(result.final_candidates)
        evidence = [_evidence_item(item) for item in build.selected]
        citations = [_citation(item) for item in build.selected]

        if self._should_rule_abstain(evidence_count=len(evidence)):
            return ChatResponse(
                answer="해당 내용은 제공된 문서 근거로 확인 불가",
                citations=citations,
                evidence=evidence,
                abstained=True,
                abstain_source="rule",
                abstain_reason="insufficient_evidence",
            )

        user_prompt = self._render_user_prompt(question=query, context=build.context)
        llm_payload = self._llm_client.generate(system_prompt=self._system_prompt, user_prompt=user_prompt)

        if self._policy.llm_abstain_enabled and bool(llm_payload.get("needs_abstain", False)):
            return ChatResponse(
                answer="해당 내용은 제공된 문서 근거로 확인 불가",
                citations=citations,
                evidence=evidence,
                abstained=True,
                abstain_source="llm",
                abstain_reason=str(llm_payload.get("reason", "llm_abstain")),
            )

        return ChatResponse(
            answer=str(llm_payload.get("answer", "")).strip(),
            citations=citations,
            evidence=evidence,
            abstained=False,
            abstain_source="none",
            abstain_reason=None,
        )

    def _render_user_prompt(self, *, question: str, context: str) -> str:
        user_block = self._user_template.format(question=question)
        if context:
            return f"{user_block}\n\n{self._context_intro}\n{context}".strip()
        return user_block

    def _should_rule_abstain(self, *, evidence_count: int) -> bool:
        if not self._policy.abstain_if_no_evidence:
            return False
        return evidence_count < self._policy.min_cited_chunks


def _citation(candidate: object) -> Citation:
    chunk = candidate.chunk
    return Citation(
        doc_id=chunk.doc_id,
        section_path=chunk.section_path,
        page_start=chunk.page_start,
        page_end=chunk.page_end,
        chunk_id=chunk.chunk_id,
    )


def _evidence_item(candidate: object) -> EvidenceItem:
    chunk = candidate.chunk
    return EvidenceItem(
        doc_id=chunk.doc_id,
        section_path=chunk.section_path,
        page_start=chunk.page_start,
        page_end=chunk.page_end,
        chunk_id=chunk.chunk_id,
        retrieval_score=candidate.retrieval_score,
        rerank_score=candidate.rerank_score,
        text=chunk.text,
    )
