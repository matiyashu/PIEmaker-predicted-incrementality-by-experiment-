"""
File-storage abstraction (Prompt 1.1).

Phase-1 implementation: local filesystem. Production swap-in: S3-compatible
backend via boto3 — same interface, switched on `S3_ENDPOINT_URL` env var.
"""

from __future__ import annotations

import io
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from app.config import get_settings

_settings = get_settings()
_UPLOAD_ROOT = Path(__file__).resolve().parents[1] / "uploads"
_UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)


@dataclass
class UploadRecord:
    upload_id: str
    filename: str
    path: str
    rows: int
    columns: list[str]


def _next_id() -> str:
    return uuid.uuid4().hex[:12]


def _store_local(content: bytes, filename: str, upload_id: str) -> Path:
    safe = Path(filename).name
    target = _UPLOAD_ROOT / upload_id
    target.mkdir(parents=True, exist_ok=True)
    path = target / safe
    path.write_bytes(content)
    return path


def parse_table(content: bytes, filename: str) -> pd.DataFrame:
    """Parse CSV / Excel into a DataFrame. Dates auto-parsed where possible."""
    suffix = Path(filename).suffix.lower()
    if suffix in (".xlsx", ".xls"):
        return pd.read_excel(io.BytesIO(content))
    return pd.read_csv(
        io.BytesIO(content),
        parse_dates=["start_date", "end_date"],
    )


def store_upload(content: bytes, filename: str) -> UploadRecord:
    upload_id = _next_id()
    path = _store_local(content, filename, upload_id)
    df = parse_table(content, filename)
    return UploadRecord(
        upload_id=upload_id,
        filename=Path(filename).name,
        path=str(path),
        rows=len(df),
        columns=df.columns.tolist(),
    )


def load_upload(upload_id: str) -> tuple[pd.DataFrame, str]:
    """Re-load a stored upload by ID. Returns (df, original_filename)."""
    folder = _UPLOAD_ROOT / upload_id
    if not folder.exists():
        raise FileNotFoundError(f"upload {upload_id} not found")
    files = list(folder.iterdir())
    if not files:
        raise FileNotFoundError(f"upload {upload_id} is empty")
    path = files[0]
    df = parse_table(path.read_bytes(), path.name)
    return df, path.name


def purge_upload(upload_id: str) -> None:
    folder = _UPLOAD_ROOT / upload_id
    if folder.exists():
        shutil.rmtree(folder)
