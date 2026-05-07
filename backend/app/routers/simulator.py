"""POST /api/simulator/run — Phase 4.2 (final)."""

from __future__ import annotations

from typing import Any, Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from services.simulator_service import SimulatorError, simulate
from services.upload_storage import load_upload

router = APIRouter(prefix="/api/simulator", tags=["simulator"])

RiskFloor = Literal["low", "unknown", "medium", "high", "severe"]


class SimulateRequest(BaseModel):
    upload_id: str | None = None
    rows: list[dict[str, Any]] | None = None
    model_id: str | None = None
    feature_set_version: str = "v1"
    cap_multiplier: float = Field(default=2.0, gt=0.0)
    total_budget_override: float | None = Field(default=None, ge=0.0)
    risk_floor: RiskFloor = "low"
    only_non_rct: bool = True


@router.post("/run")
def run(req: SimulateRequest) -> dict:
    if req.rows is not None:
        rows = req.rows
    elif req.upload_id is not None:
        try:
            df, _ = load_upload(req.upload_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc))
        if req.only_non_rct and "is_rct" in df.columns:
            df = df[df["is_rct"] != 1].copy()
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
            detail="no rows to simulate (upload empty or filtered to zero)",
        )
    try:
        return simulate(
            rows,
            model_id=req.model_id,
            feature_set_version=req.feature_set_version,
            cap_multiplier=req.cap_multiplier,
            total_budget_override=req.total_budget_override,
            risk_floor=req.risk_floor,
        )
    except SimulatorError as exc:
        # 409 — the simulator's gate has flipped (research model, etc.)
        raise HTTPException(status_code=409, detail=str(exc))
