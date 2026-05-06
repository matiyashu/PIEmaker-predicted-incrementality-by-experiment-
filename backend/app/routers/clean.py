"""POST /api/clean — Phase 1, Prompt 1.2."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from services.cleaning_service import run_pipeline
from services.mechanical_correlation_defense import decide_for_dataframe
from services.upload_storage import load_upload

router = APIRouter(prefix="/api/clean", tags=["clean"])


class CleanRequest(BaseModel):
    upload_id: str
    column_mapping: dict[str, str] | None = None
    fx_rates: dict[str, float] | None = None
    enable_winsorize: bool = False
    applied_by: str | None = None


@router.post("")
def clean(req: CleanRequest) -> dict:
    try:
        df, _ = load_upload(req.upload_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    if req.column_mapping:
        df = df.rename(columns=req.column_mapping)

    cleaned, actions = run_pipeline(
        df,
        applied_by=req.applied_by,
        fx_rates=req.fx_rates,
        enable_winsorize=req.enable_winsorize,
    )
    cleaned_with_mc = decide_for_dataframe(cleaned)

    return {
        "upload_id": req.upload_id,
        "rows_in": len(df),
        "rows_out": len(cleaned_with_mc),
        "cleaning_actions": [a.to_dict() for a in actions],
        "mc_defense": [
            d for d in cleaned_with_mc["mc_defense"].tolist() if d is not None
        ],
    }
