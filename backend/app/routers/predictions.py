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
    score_portfolio,
)
from services.upload_storage import load_upload

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


class PortfolioRequest(BaseModel):
    upload_id: str | None = None
    rows: list[dict[str, Any]] | None = None
    model_id: str | None = None
    feature_set_version: str = "v1"
    only_non_rct: bool = True


@router.post("/score-portfolio")
def score_portfolio_endpoint(req: PortfolioRequest) -> dict:
    """Score every campaign in an upload (default: non-RCT rows) or in an
    explicit `rows` payload. One model is selected per call; aggregates are
    returned alongside per-row predictions."""
    rows: list[dict]
    if req.rows is not None:
        rows = req.rows
    elif req.upload_id is not None:
        try:
            df, _ = load_upload(req.upload_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc))
        if req.only_non_rct and "is_rct" in df.columns:
            df = df[df["is_rct"] != 1].copy()
        # Coerce datetimes to ISO strings so the JSON shim can persist them.
        for col in ("start_date", "end_date"):
            if col in df.columns:
                df[col] = df[col].astype(str)
        rows = df.to_dict(orient="records")
    else:
        raise HTTPException(
            status_code=400,
            detail="provide upload_id or rows",
        )
    if not rows:
        raise HTTPException(
            status_code=400,
            detail="no rows to score (upload empty or filtered to zero)",
        )
    try:
        return score_portfolio(
            rows,
            model_id=req.model_id,
            feature_set_version=req.feature_set_version,
        )
    except PredictionError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
