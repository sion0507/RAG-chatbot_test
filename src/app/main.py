"""FastAPI entrypoint for local RAG chatbot."""

from __future__ import annotations

from fastapi import FastAPI

from src.app.routes.chat import router as chat_router
from src.app.routes.health import router as health_router


def create_app() -> FastAPI:
    """Create and configure FastAPI application."""
    app = FastAPI(title="rag-chatbot", version="0.1.0")
    app.include_router(health_router)
    app.include_router(chat_router)
    return app


app = create_app()
