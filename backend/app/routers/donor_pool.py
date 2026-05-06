"""GET/POST /api/donor-pool — Phase 2.1."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from services.donor_pool_service import (
    aging_indicator,
    coverage_heatmap,
    demote_rct,
    get_pool_size_status,
    list_eligible_rcts,
    promote_rct,
    recommend_shadow_rcts,
)
from services.upload_storage import load_upload

router = APIRouter(prefix="/api/donor-pool", tags=["donor-pool"])


def _rcts_from_upload(upload_id: str) -> list[dict]:
    df, _ = load_upload(upload_id)
    rct = df[df["is_rct"] == 1].copy()
    rct["duration_days"] = (rct["end_date"] - rct["start_date"]).dt.days
    rct["end_date"] = rct["end_date"].dt.date.astype(str)
    rct["start_date"] = rct["start_date"].dt.date.astype(str)
    return rct.to_dict(orient="records")


@router.get("/status")
def status() -> dict:
    return get_pool_size_status().to_dict()


class UploadRequest(BaseModel):
    upload_id: str


@router.post("/eligible")
def eligible(req: UploadRequest) -> dict:
    """List RCT rows from the upload with quality scores + admission flags."""
    try:
        rcts = _rcts_from_upload(req.upload_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    return {"upload_id": req.upload_id, "rcts": list_eligible_rcts(rcts)}


class PromoteRequest(BaseModel):
    upload_id: str
    campaign_ids: list[str]


@router.post("/promote")
def promote(req: PromoteRequest) -> dict:
    try:
        rcts = _rcts_from_upload(req.upload_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    by_id = {r["campaign_id"]: r for r in rcts}
    promoted = []
    for cid in req.campaign_ids:
        if cid not in by_id:
            raise HTTPException(
                status_code=400,
                detail=f"campaign_id {cid} not present in upload {req.upload_id}",
            )
        promoted.append(promote_rct(cid, by_id[cid]))
    return {
        "promoted": [r["campaign_id"] for r in promoted],
        "status": get_pool_size_status().to_dict(),
    }


class DemoteRequest(BaseModel):
    campaign_id: str


@router.post("/demote")
def demote(req: DemoteRequest) -> dict:
    record = demote_rct(req.campaign_id)
    if record is None:
        raise HTTPException(status_code=404, detail="campaign_id not in donor pool")
    return {"demoted": record, "status": get_pool_size_status().to_dict()}


@router.post("/coverage")
def coverage(req: UploadRequest) -> dict:
    try:
        rcts = _rcts_from_upload(req.upload_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    return coverage_heatmap(rcts)


@router.post("/aging")
def aging(req: UploadRequest) -> dict:
    try:
        rcts = _rcts_from_upload(req.upload_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    return aging_indicator(rcts)


class ShadowRequest(BaseModel):
    upload_id: str
    gap_threshold: int = 1


@router.post("/shadow-rcts")
def shadow(req: ShadowRequest) -> dict:
    try:
        rcts = _rcts_from_upload(req.upload_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    return {
        "recommendations": recommend_shadow_rcts(rcts, req.gap_threshold),
    }
