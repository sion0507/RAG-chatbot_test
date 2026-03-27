"""Health route."""

from __future__ import annotations

from fastapi import APIRouter

from src.app.schemas import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Return static readiness payload."""
    return HealthResponse(
        status="ok",
        components={
            "api": "ready",
        },
    )
