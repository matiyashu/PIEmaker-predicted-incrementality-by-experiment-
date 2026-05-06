"""POST /api/features/build — Phase 2.2."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from services.feature_engineering_service import (
    DEFAULT_FEATURE_SET_VERSION,
    X_POST_FIELDS,
    X_PRE_FIELDS,
    build_features,
)
from services.upload_storage import load_upload

router = APIRouter(prefix="/api/features", tags=["features"])


class BuildRequest(BaseModel):
    upload_id: str
    mode: str = "training"
    feature_set_version: str = DEFAULT_FEATURE_SET_VERSION
    sample_id: str | None = None


@router.post("/build")
def build(req: BuildRequest) -> dict:
    if req.mode not in ("training", "scoring"):
        raise HTTPException(status_code=400, detail="mode must be 'training' or 'scoring'")
    try:
        df, _ = load_upload(req.upload_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    feats_df = build_features(
        df,
        mode=req.mode,                                         # type: ignore[arg-type]
        feature_set_version=req.feature_set_version,
        sample_id=req.sample_id,
    )
    return {
        "upload_id": req.upload_id,
        "mode": req.mode,
        "feature_set_version": req.feature_set_version,
        "x_pre_fields": X_PRE_FIELDS,
        "x_post_fields": X_POST_FIELDS,
        "rows": feats_df.to_dict(orient="records"),
    }
