"""POST /api/schema/map — Phase 1, Prompt 1.1."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from services.schema_mapping import (
    apply_mapping,
    suggest_mappings,
    to_dict_list,
)
from services.upload_storage import load_upload

router = APIRouter(prefix="/api/schema", tags=["schema"])


class MapSuggestRequest(BaseModel):
    upload_id: str


class MapApplyRequest(BaseModel):
    upload_id: str
    mapping: dict[str, str]


@router.post("/suggest")
def suggest(req: MapSuggestRequest) -> dict:
    try:
        df, _ = load_upload(req.upload_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    suggestions = suggest_mappings(df.columns.tolist())
    return {
        "upload_id": req.upload_id,
        "source_columns": df.columns.tolist(),
        "suggestions": to_dict_list(suggestions),
    }


@router.post("/apply")
def apply(req: MapApplyRequest) -> dict:
    try:
        df, _ = load_upload(req.upload_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    try:
        normalized = apply_mapping(df.columns.tolist(), req.mapping)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return {"upload_id": req.upload_id, "mapping": normalized}
