"""Chat route."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from src.app.dependencies import get_chat_service
from src.app.schemas import ChatRequest, ChatResponse
from src.app.services.chat_service import ChatService

router = APIRouter(tags=["chat"])


@router.post("/chat", response_model=ChatResponse)
def chat(
    request: ChatRequest,
    service: ChatService = Depends(get_chat_service),
) -> ChatResponse:
    """Execute retrieval-grounded answering flow."""
    try:
        return service.ask(question=request.question, top_n=request.top_n)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"chat 처리 실패: {exc}") from exc
