"""POST /api/models/* — training, ablation, hold-out, calibration, advertiser CV.

Phase 2.3 + 2.4 surfaces shipped in V.3. V.4 Wave 2 adds:
  * POST /api/models/calibration         — LCC calibration by segment
  * POST /api/models/sample-size-curve   — paper Figure 6
  * POST /api/models/advertiser-cv       — existing vs new advertiser cohorts
  * POST /api/models/bootstrap-advertisers — cluster bootstrap CI
  * GET  /api/models/holdout-distributions — full hold-out R² distributions
"""

from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ml.advertiser_cv import run_existing_vs_new
from ml.bootstrap_advertisers import bootstrap_over_advertisers
from ml.holdout_one_level import SEGMENTATION_VARS, run_extrapolation_test
from ml.lcc_calibration_by_segment import calibration_by_segment
from ml.model_registry import list_models, metrics_for, promote_to_production
from ml.sample_size_curve import run_pool_size_curve
from services.model_training_service import train_pie_model
from services.persistence import read_table, upsert

HOLDOUT_RESULTS_TABLE = "holdout_results"
HOLDOUT_DISTRIBUTIONS_TABLE = "holdout_distributions"


def _load_paper_pool(feature_set_version: str) -> tuple[list[dict], list[dict]]:
    feats = [
        r
        for r in read_table("feature_store")
        if r.get("feature_set_version") == feature_set_version
        and r.get("mode") == "training"
    ]
    labels = [
        r for r in read_table("rct_labels") if r.get("icpd") is not None
    ]
    if not feats or not labels:
        raise HTTPException(
            status_code=400,
            detail="no feature_store / rct_labels rows. Build features first.",
        )
    return feats, labels


def _cost_weights(feats: list[dict]) -> list[float]:
    out: list[float] = []
    for f in feats:
        c = f.get("cost")
        if c is None:
            out.append(1.0)
        else:
            try:
                out.append(float(c))
            except (TypeError, ValueError):
                out.append(1.0)
    return out

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
    # V.4 Wave 2: legacy `holdout_results` keeps median-only payload so the
    # prediction service stays backwards-compatible; the new
    # `holdout_distributions` table carries the full R² distributions for
    # the diagnostics UI.
    now = datetime.now(timezone.utc).isoformat()
    for row in results:
        key_id = f"{row['segmentation_var']}|{row['level']}"
        median_only = {
            k: v
            for k, v in row.items()
            if k
            not in {
                "within_r2_dist",
                "extrapolation_r2_dist",
                "penalty_pp_dist",
            }
        }
        upsert(
            HOLDOUT_RESULTS_TABLE,
            {**median_only, "id": key_id, "created_at": now},
            key="id",
        )
        upsert(
            HOLDOUT_DISTRIBUTIONS_TABLE,
            {**row, "id": key_id, "created_at": now},
            key="id",
        )

    return {
        "segmentation_var": req.segmentation_var,
        "n_iterations": req.n_iterations,
        "results": results,
    }


# --- V.4 Wave 2 new endpoints --------------------------------------------------


@router.get("/holdout-distributions")
def list_holdout_distributions(segmentation_var: str | None = None) -> dict:
    rows = read_table(HOLDOUT_DISTRIBUTIONS_TABLE)
    if segmentation_var:
        rows = [r for r in rows if r.get("segmentation_var") == segmentation_var]
    return {"distributions": rows, "count": len(rows)}


class CalibrationRequest(BaseModel):
    feature_set_version: str = "v1"
    segmentation_var: str
    min_n_per_level: int = 5


@router.post("/calibration")
def calibration(req: CalibrationRequest) -> dict:
    feats, labels = _load_paper_pool(req.feature_set_version)
    try:
        rows = calibration_by_segment(
            feature_rows=feats,
            label_rows=labels,
            segmentation_var=req.segmentation_var,
            cost_weights=_cost_weights(feats),
            min_n_per_level=req.min_n_per_level,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return {
        "segmentation_var": req.segmentation_var,
        "n_levels": len(rows),
        "results": rows,
    }


class SampleSizeRequest(BaseModel):
    feature_set_version: str = "v1"
    sizes: list[int] | None = None
    n_subsamples: int = 5
    n_splits: int = 5
    seed: int = 42


@router.post("/sample-size-curve")
def sample_size_curve(req: SampleSizeRequest) -> dict:
    feats, labels = _load_paper_pool(req.feature_set_version)
    try:
        points = run_pool_size_curve(
            feature_rows=feats,
            label_rows=labels,
            cost_weights=_cost_weights(feats),
            sizes=req.sizes,
            n_subsamples=req.n_subsamples,
            n_splits=req.n_splits,
            seed=req.seed,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return {"n_points": len(points), "points": points}


class AdvertiserCvRequest(BaseModel):
    feature_set_version: str = "v1"
    n_splits: int = 5
    advertiser_id_col: str = "advertiser_id"


@router.post("/advertiser-cv")
def advertiser_cv(req: AdvertiserCvRequest) -> dict:
    feats, labels = _load_paper_pool(req.feature_set_version)
    try:
        return run_existing_vs_new(
            feature_rows=feats,
            label_rows=labels,
            cost_weights=_cost_weights(feats),
            advertiser_id_col=req.advertiser_id_col,
            n_splits=req.n_splits,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


class AdvertiserBootstrapRequest(BaseModel):
    feature_set_version: str = "v1"
    n_draws: int = 100
    advertiser_id_col: str = "advertiser_id"
    seed: int = 42


@router.post("/bootstrap-advertisers")
def bootstrap_advertisers(req: AdvertiserBootstrapRequest) -> dict:
    feats, labels = _load_paper_pool(req.feature_set_version)
    try:
        return bootstrap_over_advertisers(
            feature_rows=feats,
            label_rows=labels,
            cost_weights=_cost_weights(feats),
            advertiser_id_col=req.advertiser_id_col,
            n_draws=req.n_draws,
            seed=req.seed,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
