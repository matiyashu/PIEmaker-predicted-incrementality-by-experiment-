"""POST /api/drift/check — Phase 4.1."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from services.drift_service import compute_drift
from services.feature_engineering_service import build_features
from services.upload_storage import load_upload

router = APIRouter(prefix="/api/drift", tags=["drift"])


class DriftRequest(BaseModel):
    upload_id: str
    feature_set_version: str = "v1"
    only_non_rct: bool = True


@router.post("/check")
def check(req: DriftRequest) -> dict:
    """Compare an upload's feature distribution to the trained feature_store.

    Builds scoring features in-memory (without polluting feature_store via
    upserts is done already by build_features — they're tagged mode=scoring
    so the trainer filters them out) and runs PSI per column.
    """
    try:
        df, _ = load_upload(req.upload_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    if req.only_non_rct and "is_rct" in df.columns:
        df = df[df["is_rct"] != 1].copy()
    if df.empty:
        raise HTTPException(
            status_code=400,
            detail="upload has no rows after filtering (is_rct != 1)",
        )

    feats_df = build_features(
        df,
        mode="scoring",
        feature_set_version=req.feature_set_version,
        sample_id=None,
    )
    scoring_rows = feats_df.to_dict(orient="records")

    try:
        return compute_drift(
            scoring_rows,
            feature_set_version=req.feature_set_version,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
