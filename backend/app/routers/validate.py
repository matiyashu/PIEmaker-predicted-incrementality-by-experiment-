"""POST /api/validate — Phase 1, Prompt 1.1."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from services.upload_storage import load_upload
from services.validation_service import validate_dataframe

router = APIRouter(prefix="/api/validate", tags=["validate"])


class ValidateRequest(BaseModel):
    upload_id: str
    column_mapping: dict[str, str] | None = None


@router.post("")
def validate(req: ValidateRequest) -> dict:
    try:
        df, _ = load_upload(req.upload_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    if req.column_mapping:
        df = df.rename(columns=req.column_mapping)
    result = validate_dataframe(df)
    return {"upload_id": req.upload_id, **result.to_dict()}
