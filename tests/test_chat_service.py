from __future__ import annotations

import unittest

from src.app.services.chat_service import ChatService, GroundingPolicy
from src.app.services.context_builder import ContextBuilder
from src.embed.store import ChunkRecord
from src.rag.retriever import FinalCandidate, RetrievalResult


class _FakeRetriever:
    def __init__(self, final_candidates: list[FinalCandidate]) -> None:
        self._final_candidates = final_candidates

    def retrieve(self, query: str, *, reranker: object, final_top_n: int) -> RetrievalResult:
        _ = (query, reranker, final_top_n)
        return RetrievalResult(
            query=query,
            candidates=[],
            rerank_candidates=[],
            final_candidates=self._final_candidates,
        )


class _FakeLlm:
    def __init__(self, payload: dict) -> None:
        self.payload = payload

    def generate(self, *, system_prompt: str, user_prompt: str) -> dict:
        _ = (system_prompt, user_prompt)
        return self.payload


class ChatServiceTest(unittest.TestCase):
    def _final(self, chunk_id: str, text: str) -> FinalCandidate:
        return FinalCandidate(
            chunk=ChunkRecord(
                doc_id="manual",
                section_path="2.절차",
                heading="절차",
                chunk_id=chunk_id,
                chunk_index=0,
                page_start=2,
                page_end=2,
                text=text,
                chunk_tokens=3,
            ),
            retrieval_score=0.8,
            rerank_score=1.1,
        )

    def test_rule_based_abstain_when_no_evidence(self) -> None:
        service = ChatService(
            retriever=_FakeRetriever(final_candidates=[]),
            reranker=object(),
            llm_client=_FakeLlm(payload={"answer": "무시", "needs_abstain": False, "reason": ""}),
            context_builder=ContextBuilder(
                max_context_chars=1000,
                max_chunks_in_context=3,
                include_chunk_header=True,
            ),
            system_prompt="sys",
            user_template="질문: {question}",
            context_intro="근거",
            policy=GroundingPolicy(abstain_if_no_evidence=True, min_cited_chunks=1, llm_abstain_enabled=True),
            final_top_n=3,
        )

        result = service.ask("질문")
        self.assertTrue(result.abstained)
        self.assertEqual(result.abstain_source, "rule")

    def test_llm_abstain_signal_is_applied(self) -> None:
        service = ChatService(
            retriever=_FakeRetriever(final_candidates=[self._final("manual:1", "근거 텍스트")]),
            reranker=object(),
            llm_client=_FakeLlm(payload={"answer": "", "needs_abstain": True, "reason": "ambiguous"}),
            context_builder=ContextBuilder(
                max_context_chars=1000,
                max_chunks_in_context=3,
                include_chunk_header=True,
            ),
            system_prompt="sys",
            user_template="질문: {question}",
            context_intro="근거",
            policy=GroundingPolicy(abstain_if_no_evidence=True, min_cited_chunks=1, llm_abstain_enabled=True),
            final_top_n=3,
        )

        result = service.ask("질문")
        self.assertTrue(result.abstained)
        self.assertEqual(result.abstain_source, "llm")

    def test_success_response_includes_evidence_and_citations(self) -> None:
        service = ChatService(
            retriever=_FakeRetriever(final_candidates=[self._final("manual:1", "근거 텍스트")]),
            reranker=object(),
            llm_client=_FakeLlm(payload={"answer": "점검해야 합니다", "needs_abstain": False, "reason": ""}),
            context_builder=ContextBuilder(
                max_context_chars=1000,
                max_chunks_in_context=3,
                include_chunk_header=True,
            ),
            system_prompt="sys",
            user_template="질문: {question}",
            context_intro="근거",
            policy=GroundingPolicy(abstain_if_no_evidence=True, min_cited_chunks=1, llm_abstain_enabled=True),
            final_top_n=3,
        )

        result = service.ask("질문")
        self.assertFalse(result.abstained)
        self.assertEqual(result.answer, "점검해야 합니다")
        self.assertEqual(len(result.citations), 1)
        self.assertEqual(len(result.evidence), 1)


if __name__ == "__main__":
    unittest.main()
