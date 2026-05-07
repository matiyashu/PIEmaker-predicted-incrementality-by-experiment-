"""POST /api/decisions/recommend — Phase 3.3."""

from __future__ import annotations

from typing import Any, Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from services.decision_service import (
    project_portfolio_lift,
    rank_recommendations,
)
from services.prediction_service import (
    PredictionError,
    score_portfolio,
)
from services.upload_storage import load_upload

router = APIRouter(prefix="/api/decisions", tags=["decisions"])

RiskFloor = Literal["low", "unknown", "medium", "high", "severe"]


class RecommendRequest(BaseModel):
    upload_id: str | None = None
    rows: list[dict[str, Any]] | None = None
    runs: list[dict[str, Any]] | None = None
    model_id: str | None = None
    feature_set_version: str = "v1"
    only_non_rct: bool = True
    risk_floor: RiskFloor = "low"


@router.post("/recommend")
def recommend(req: RecommendRequest) -> dict:
    """Rank a portfolio (or pre-scored runs) into advisory action bands."""
    runs: list[dict]
    is_research = False
    portfolio_meta: dict | None = None

    if req.runs is not None:
        runs = req.runs
        statuses = {r.get("model_status") for r in runs}
        is_research = statuses == {"research"} or "research" in statuses
    elif req.upload_id is not None or req.rows is not None:
        if req.rows is not None:
            score_rows = req.rows
        else:
            try:
                df, _ = load_upload(req.upload_id)  # type: ignore[arg-type]
            except FileNotFoundError as exc:
                raise HTTPException(status_code=404, detail=str(exc))
            if req.only_non_rct and "is_rct" in df.columns:
                df = df[df["is_rct"] != 1].copy()
            for col in ("start_date", "end_date"):
                if col in df.columns:
                    df[col] = df[col].astype(str)
            score_rows = df.to_dict(orient="records")
        if not score_rows:
            raise HTTPException(
                status_code=400,
                detail="no rows to score (upload empty or filtered to zero)",
            )
        try:
            portfolio = score_portfolio(
                score_rows,
                model_id=req.model_id,
                feature_set_version=req.feature_set_version,
            )
        except PredictionError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        runs = portfolio["runs"]
        is_research = portfolio["model"]["status"] == "research"
        portfolio_meta = {
            "model": portfolio["model"],
            "aggregates": portfolio["aggregates"],
            "watermark": portfolio["watermark"],
        }
    else:
        raise HTTPException(
            status_code=400,
            detail="provide upload_id, rows, or runs",
        )

    recs = rank_recommendations(runs, risk_floor=req.risk_floor)
    lift = project_portfolio_lift(recs, is_research_model=is_research)

    counts: dict[str, int] = {
        "promote": 0,
        "hold": 0,
        "deprioritize": 0,
        "block": 0,
    }
    for r in recs:
        counts[r["action"]] = counts.get(r["action"], 0) + 1

    return {
        "recommendations": recs,
        "action_counts": counts,
        "projected_lift": lift,
        "is_research_model": is_research,
        "risk_floor": req.risk_floor,
        "portfolio": portfolio_meta,
    }
