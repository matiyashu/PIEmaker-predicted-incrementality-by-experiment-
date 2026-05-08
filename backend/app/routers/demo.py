"""POST /api/demo/seed, GET /api/demo/status — demo bootstrapping."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from services.demo_seed_service import DemoSeedError, get_status, seed

router = APIRouter(prefix="/api/demo", tags=["demo"])


@router.get("/status")
def status() -> dict:
    return get_status()


@router.post("/seed")
def seed_endpoint() -> dict:
    try:
        return seed()
    except DemoSeedError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
