"""POST /api/predictions/score, GET /api/predictions, GET /api/predictions/{id} — Phase 3.1."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from services.prediction_service import (
    PredictionError,
    get_run,
    list_runs,
    score_campaign,
)

router = APIRouter(prefix="/api/predictions", tags=["predictions"])


class ScoreRequest(BaseModel):
    spec: dict[str, Any]
    model_id: str | None = None
    feature_set_version: str = "v1"


@router.post("/score")
def score(req: ScoreRequest) -> dict:
    try:
        return score_campaign(
            req.spec,
            model_id=req.model_id,
            feature_set_version=req.feature_set_version,
        )
    except PredictionError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.get("")
def list_all(limit: int = 50) -> dict:
    return {"runs": list_runs(limit=limit)}


@router.get("/{run_id}")
def get(run_id: str) -> dict:
    run = get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"run {run_id} not found")
    return run
