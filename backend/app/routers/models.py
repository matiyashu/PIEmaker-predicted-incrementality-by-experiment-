"""POST /api/models/train, /api/models/evaluate, /api/models/holdout-one-level — Phase 2.3 + 2.4."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from datetime import datetime, timezone

from ml.holdout_one_level import SEGMENTATION_VARS, run_extrapolation_test
from ml.model_registry import list_models, metrics_for, promote_to_production
from services.model_training_service import train_pie_model
from services.persistence import read_table, upsert

HOLDOUT_RESULTS_TABLE = "holdout_results"

router = APIRouter(prefix="/api/models", tags=["models"])


class TrainRequest(BaseModel):
    feature_set_version: str = "v1"
    name: str = "pie_random_forest"
    n_bootstrap: int = 200


@router.post("/train")
def train(req: TrainRequest) -> dict:
    try:
        return train_pie_model(
            feature_set_version=req.feature_set_version,
            name=req.name,
            n_bootstrap=req.n_bootstrap,
        )
    except PermissionError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.get("")
def list_all(status: str | None = None) -> dict:
    return {"models": list_models(status=status)}


@router.get("/{model_id}/metrics")
def metrics(model_id: str) -> dict:
    return {"model_id": model_id, "metrics": metrics_for(model_id)}


class PromoteRequest(BaseModel):
    model_id: str


@router.post("/promote")
def promote(req: PromoteRequest) -> dict:
    try:
        return promote_to_production(req.model_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


class HoldoutRequest(BaseModel):
    feature_set_version: str = "v1"
    segmentation_var: str
    n_iterations: int = 50


@router.post("/holdout-one-level")
def holdout(req: HoldoutRequest) -> dict:
    if req.segmentation_var not in SEGMENTATION_VARS:
        raise HTTPException(
            status_code=400,
            detail=f"segmentation_var must be one of {list(SEGMENTATION_VARS)}",
        )
    feats = [
        r
        for r in read_table("feature_store")
        if r.get("feature_set_version") == req.feature_set_version
        and r.get("mode") == "training"
    ]
    labels = read_table("rct_labels")
    try:
        results = run_extrapolation_test(
            feature_rows=feats,
            label_rows=labels,
            segmentation_var=req.segmentation_var,
            n_iterations=req.n_iterations,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    # Persist per-level results so the prediction service can attach
    # extrapolation-risk badges to forecasts (Phase 3.1).
    now = datetime.now(timezone.utc).isoformat()
    for row in results:
        key_id = f"{row['segmentation_var']}|{row['level']}"
        upsert(
            HOLDOUT_RESULTS_TABLE,
            {**row, "id": key_id, "created_at": now},
            key="id",
        )

    return {
        "segmentation_var": req.segmentation_var,
        "n_iterations": req.n_iterations,
        "results": results,
    }
