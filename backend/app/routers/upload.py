"""POST /api/upload — Phase 1, Prompt 1.1."""

from __future__ import annotations

from fastapi import APIRouter, File, HTTPException, UploadFile

from services.upload_storage import store_upload

router = APIRouter(prefix="/api/upload", tags=["upload"])

MAX_BYTES = 100 * 1024 * 1024  # 100 MB


@router.post("")
async def upload_file(file: UploadFile = File(...)) -> dict:
    if not file.filename:
        raise HTTPException(status_code=400, detail="filename required")
    suffix = file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
    if suffix not in {"csv", "xlsx", "xls"}:
        raise HTTPException(
            status_code=400,
            detail="only .csv, .xlsx, .xls are supported",
        )
    content = await file.read()
    if len(content) > MAX_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"file exceeds {MAX_BYTES // (1024 * 1024)} MB limit",
        )
    record = store_upload(content, file.filename)
    return {
        "upload_id": record.upload_id,
        "filename": record.filename,
        "rows": record.rows,
        "columns": record.columns,
    }
