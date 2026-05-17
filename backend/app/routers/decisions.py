"""POST /api/decisions/* — recommend (Phase 3.3) + curves (V.4 Wave 3 / Phase 5)."""

from __future__ import annotations

from typing import Any, Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from pie_formulas.decision_curves import (
    disagreement_curves,
    disagreement_curves_compare,
    expected_disagreement_cost_curve,
)
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


# --- V.4 Wave 3 (Phase 5) — disagreement curves ------------------------------


class DisagreementCurvesRequest(BaseModel):
    """One curve, one model.

    `icpd_true` and `icpd_pred` are the paper-faithful inputs — the
    diagnostic doesn't know how the predictions were generated. Callers
    typically pull these from the orchestrator's OOF arrays.
    """

    icpd_true: list[float]
    icpd_pred: list[float]
    reference_median: float | None = None
    low_ratio: float = 0.5
    high_ratio: float = 1.5
    step: float = 0.05
    # PIEmaker extension: when both per-unit costs are >= 0, an
    # expected_cost field is added to each curve point. Clearly labelled
    # as a PIEmaker extension in the response so it can't be confused
    # with the paper's plain D(t).
    fp_cost_per_unit: float | None = None
    fn_cost_per_unit: float | None = None


@router.post("/curves")
def curves(req: DisagreementCurvesRequest) -> dict:
    if len(req.icpd_true) != len(req.icpd_pred):
        raise HTTPException(
            status_code=400,
            detail="icpd_true and icpd_pred must have the same length",
        )
    try:
        curve = disagreement_curves(
            icpd_true=req.icpd_true,
            icpd_pred=req.icpd_pred,
            reference_median=req.reference_median,
            low_ratio=req.low_ratio,
            high_ratio=req.high_ratio,
            step=req.step,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    edc = None
    if req.fp_cost_per_unit is not None and req.fn_cost_per_unit is not None:
        try:
            edc = expected_disagreement_cost_curve(
                curve,
                fp_cost_per_unit=req.fp_cost_per_unit,
                fn_cost_per_unit=req.fn_cost_per_unit,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    return {
        "curve": curve,
        "expected_cost_curve": edc,
        "reference_median": req.reference_median,
        "low_ratio": req.low_ratio,
        "high_ratio": req.high_ratio,
        "step": req.step,
    }


class DisagreementCompareRequest(BaseModel):
    """PIE vs Raw-LCC-7D side-by-side over the same threshold sweep."""

    icpd_true: list[float]
    pie_pred: list[float]
    raw_lcc_pred: list[float]
    reference_median: float | None = None
    low_ratio: float = 0.5
    high_ratio: float = 1.5
    step: float = 0.05


@router.post("/curves/compare")
def curves_compare(req: DisagreementCompareRequest) -> dict:
    n = len(req.icpd_true)
    if len(req.pie_pred) != n or len(req.raw_lcc_pred) != n:
        raise HTTPException(
            status_code=400,
            detail=(
                "icpd_true, pie_pred, and raw_lcc_pred must all have the "
                "same length"
            ),
        )
    try:
        return disagreement_curves_compare(
            icpd_true=req.icpd_true,
            pie_pred=req.pie_pred,
            raw_lcc_pred=req.raw_lcc_pred,
            reference_median=req.reference_median,
            low_ratio=req.low_ratio,
            high_ratio=req.high_ratio,
            step=req.step,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
