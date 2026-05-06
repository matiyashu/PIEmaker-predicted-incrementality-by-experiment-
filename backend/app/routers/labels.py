"""POST /api/labels/generate — Phase 2.2."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from services.label_generation_service import generate_labels
from services.upload_storage import load_upload

router = APIRouter(prefix="/api/labels", tags=["labels"])


class GenerateRequest(BaseModel):
    upload_id: str
    has_user_level_data: bool = False


@router.post("/generate")
def generate(req: GenerateRequest) -> dict:
    try:
        df, _ = load_upload(req.upload_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    rct_rows = df[df["is_rct"] == 1].copy()
    if rct_rows.empty:
        raise HTTPException(
            status_code=400,
            detail="upload contains no RCT rows (is_rct == 1)",
        )
    labels_df = generate_labels(rct_rows, has_user_level_data=req.has_user_level_data)
    return {
        "upload_id": req.upload_id,
        "labels": labels_df.to_dict(orient="records"),
    }
