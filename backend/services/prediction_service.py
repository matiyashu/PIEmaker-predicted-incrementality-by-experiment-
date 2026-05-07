"""
Prediction service (Prompt 3.1).

Scores a single campaign spec against a registered model. Returns ICPD with
a confidence band (derived from the model's bootstrap metric) and tags each
prediction with the extrapolation-risk band that applies to the campaign's
segment levels (from Phase 2.4 hold-out-one-level results) and a watermark
when the model was trained from a research-band donor pool.

Decision logic for which model to use:
  1. If model_id is supplied, use it.
  2. Else: latest production model.
  3. Else: latest research model (response is watermarked).
  4. Else: 404.
"""

from __future__ import annotations

import statistics
import uuid
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from ml.holdout_one_level import SEGMENTATION_VARS
from ml.model_registry import (
    MODEL_VERSIONS_TABLE,
    load_model,
    metrics_for,
)
from services.feature_engineering_service import (
    DEFAULT_FEATURE_SET_VERSION,
    X_POST_FIELDS,
    X_PRE_FIELDS,
    build_features,
)
from services.persistence import find_by, read_table, upsert

PREDICTION_RUNS_TABLE = "prediction_runs"

# Fields the caller may set via /predictions/score. X_post is optional —
# pre-launch predictions only have X_pre; we let the model impute the rest.
INPUT_FIELDS = (
    *X_PRE_FIELDS,
    *X_POST_FIELDS,
    "campaign_id",
    "start_date",
    "end_date",
    "cost",
    "test_users",
    "control_users",
    "exposed_test_users",
    "clicks",
    "impressions",
    "conversions",
    "lcc_1h",
    "lcc_1d",
    "lcc_7d",
    "lcc_28d",
    "view_through_conversions",
    "avg_dwell_time",
)


class PredictionError(Exception):
    """Raised when the prediction cannot be produced (no model, bad spec, etc.)."""


def _select_model(model_id: str | None) -> dict:
    rows = read_table(MODEL_VERSIONS_TABLE)
    if not rows:
        raise PredictionError("no models registered; train one first")
    if model_id:
        match = next((r for r in rows if r["id"] == model_id), None)
        if not match:
            raise PredictionError(f"model_id {model_id} not found")
        return match
    prod = sorted(
        (r for r in rows if r.get("status") == "production"),
        key=lambda r: r["created_at"],
        reverse=True,
    )
    if prod:
        return prod[0]
    research = sorted(rows, key=lambda r: r["created_at"], reverse=True)
    return research[0]


def _ci_half_width(model_id: str) -> float | None:
    """Use the bootstrap mean ± half-width recorded at training as the CI band."""
    for m in metrics_for(model_id):
        if m["metric_type"] == "weighted_r_squared_bootstrap_mean":
            lo = m.get("ci_lower")
            hi = m.get("ci_upper")
            if lo is not None and hi is not None:
                return float(hi - lo) / 2.0
    return None


def _segment_risk(spec: dict, segmentation_var: str) -> dict | None:
    """Look up the most recent hold-out-one-level result for the spec's level.

    The /api/models/holdout-one-level endpoint persists into the
    `holdout_results` table; rows are keyed by f"{var}|{level}" so the most
    recent run for a (var, level) pair always wins.
    """
    level_value = spec.get(segmentation_var)
    if level_value is None:
        return None
    level_str = str(level_value)
    key_id = f"{segmentation_var}|{level_str}"
    row = find_by("holdout_results", "id", key_id)
    if row is None:
        return None
    return {
        "segmentation_var": segmentation_var,
        "level": level_str,
        "within_r2_median": row.get("within_r2_median"),
        "extrapolation_r2_median": row.get("extrapolation_r2_median"),
        "penalty_pp": row.get("penalty_pp"),
        "risk": row.get("risk", "unknown"),
    }


def _spec_to_frame(spec: dict) -> pd.DataFrame:
    cleaned = {k: spec.get(k) for k in INPUT_FIELDS if k in spec}
    if "campaign_id" not in cleaned:
        cleaned["campaign_id"] = f"PRED-{uuid.uuid4().hex[:8]}"
    return pd.DataFrame([cleaned])


def score_campaign(
    spec: dict[str, Any],
    model_id: str | None = None,
    feature_set_version: str = DEFAULT_FEATURE_SET_VERSION,
) -> dict:
    """Score a single campaign spec; persist a prediction_runs row; return result."""
    model = _select_model(model_id)
    estimator = load_model(model["id"])

    frame = _spec_to_frame(spec)
    cid = str(frame.iloc[0]["campaign_id"])

    # Build features in scoring mode but DON'T persist into feature_store —
    # scoring features are ephemeral. Use the underlying flatten path used by
    # the trainer so the column shape lines up.
    feats_df = build_features(
        frame,
        mode="scoring",
        feature_set_version=feature_set_version,
        sample_id=None,
    )
    pred_value = float(estimator.predict(_flatten_for_model(feats_df))[0])

    half_width = _ci_half_width(model["id"])
    ci_lower = pred_value - half_width if half_width is not None else None
    ci_upper = pred_value + half_width if half_width is not None else None

    risks = [r for v in SEGMENTATION_VARS if (r := _segment_risk(spec, v))]
    worst = max(
        (r for r in risks if r.get("penalty_pp") is not None),
        key=lambda r: r["penalty_pp"],
        default=None,
    )

    watermark = None
    if model["status"] == "research":
        watermark = (
            "Research-mode model (donor pool < 400 RCTs). Predictions are "
            "advisory only and may not be used in the Decision Simulator."
        )

    run_id = uuid.uuid4().hex[:12]
    record = {
        "id": run_id,
        "campaign_id": cid,
        "model_version_id": model["id"],
        "model_status": model["status"],
        "feature_set_version": feature_set_version,
        "predicted_icpd": pred_value,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "segment_risks": risks,
        "worst_segment_risk": worst,
        "watermark": watermark,
        "spec": spec,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    upsert(PREDICTION_RUNS_TABLE, record, key="id")
    return record


def _flatten_for_model(feats_df: pd.DataFrame) -> pd.DataFrame:
    """Match the flatten that train_random_forest._flatten_feature_rows does."""
    rows = feats_df.to_dict(orient="records")
    flat = []
    for r in rows:
        rec = {"campaign_id": r["campaign_id"]}
        rec.update(r.get("x_pre") or {})
        rec.update(r.get("x_post") or {})
        flat.append(rec)
    out = pd.DataFrame(flat)
    if "campaign_id" in out.columns:
        out = out.set_index("campaign_id")
    return out


def get_run(run_id: str) -> dict | None:
    return find_by(PREDICTION_RUNS_TABLE, "id", run_id)


def list_runs(limit: int = 50) -> list[dict]:
    rows = read_table(PREDICTION_RUNS_TABLE)
    rows.sort(key=lambda r: r["created_at"], reverse=True)
    return rows[:limit]


# --- Portfolio scoring (Prompt 3.2) ----------------------------------------


def score_portfolio(
    rows: list[dict],
    model_id: str | None = None,
    feature_set_version: str = DEFAULT_FEATURE_SET_VERSION,
) -> dict:
    """Score a list of campaign specs (a portfolio) against one model.

    Picks the model once (same precedence as score_campaign), batches the
    feature build + predict, then attaches segment-risk badges per row. Each
    row is persisted to prediction_runs so /api/predictions reflects the
    portfolio entries alongside single-shot scores.

    Returns: {
      model: <ModelRecord>,
      runs: [<PredictionRun>],
      aggregates: {n, mean_icpd, median_icpd, stdev_icpd, p10_icpd, p90_icpd,
                   risk_counts: {severe, high, medium, low, unknown}},
      worst_segment_risk: <SegmentRisk | None>,  # the worst across the portfolio
    }
    """
    if not rows:
        raise PredictionError("portfolio is empty")

    model = _select_model(model_id)
    estimator = load_model(model["id"])
    half_width = _ci_half_width(model["id"])
    watermark = (
        "Research-mode model (donor pool < 400 RCTs). Predictions are "
        "advisory only and may not be used in the Decision Simulator."
        if model["status"] == "research"
        else None
    )

    frame = pd.DataFrame(
        [
            {k: r.get(k) for k in INPUT_FIELDS if k in r}
            for r in rows
        ]
    )
    if "campaign_id" not in frame.columns:
        frame["campaign_id"] = [
            f"PRED-{uuid.uuid4().hex[:8]}" for _ in range(len(frame))
        ]
    else:
        # Fill any missing campaign_ids with a generated one.
        missing = frame["campaign_id"].isna()
        if missing.any():
            frame.loc[missing, "campaign_id"] = [
                f"PRED-{uuid.uuid4().hex[:8]}" for _ in range(int(missing.sum()))
            ]

    feats_df = build_features(
        frame,
        mode="scoring",
        feature_set_version=feature_set_version,
        sample_id=None,
    )
    preds = estimator.predict(_flatten_for_model(feats_df)).tolist()

    runs: list[dict] = []
    now = datetime.now(timezone.utc).isoformat()
    for spec, pred in zip(rows, preds, strict=True):
        cid = str(spec.get("campaign_id") or f"PRED-{uuid.uuid4().hex[:8]}")
        risks = [r for v in SEGMENTATION_VARS if (r := _segment_risk(spec, v))]
        worst = max(
            (r for r in risks if r.get("penalty_pp") is not None),
            key=lambda r: r["penalty_pp"],
            default=None,
        )
        run = {
            "id": uuid.uuid4().hex[:12],
            "campaign_id": cid,
            "model_version_id": model["id"],
            "model_status": model["status"],
            "feature_set_version": feature_set_version,
            "predicted_icpd": float(pred),
            "ci_lower": float(pred) - half_width if half_width is not None else None,
            "ci_upper": float(pred) + half_width if half_width is not None else None,
            "segment_risks": risks,
            "worst_segment_risk": worst,
            "watermark": watermark,
            "spec": spec,
            "portfolio_run": True,
            "created_at": now,
        }
        upsert(PREDICTION_RUNS_TABLE, run, key="id")
        runs.append(run)

    icpds = [r["predicted_icpd"] for r in runs]
    risk_counts = {"severe": 0, "high": 0, "medium": 0, "low": 0, "unknown": 0}
    for r in runs:
        if r["worst_segment_risk"]:
            risk_counts[r["worst_segment_risk"]["risk"]] = (
                risk_counts.get(r["worst_segment_risk"]["risk"], 0) + 1
            )
        else:
            risk_counts["unknown"] += 1

    aggregates = {
        "n": len(icpds),
        "mean_icpd": float(np.mean(icpds)),
        "median_icpd": float(np.median(icpds)),
        "stdev_icpd": float(statistics.pstdev(icpds)) if len(icpds) > 1 else 0.0,
        "p10_icpd": float(np.percentile(icpds, 10)),
        "p90_icpd": float(np.percentile(icpds, 90)),
        "risk_counts": risk_counts,
    }
    portfolio_worst = max(
        (
            r["worst_segment_risk"]
            for r in runs
            if r["worst_segment_risk"]
            and r["worst_segment_risk"].get("penalty_pp") is not None
        ),
        key=lambda x: x["penalty_pp"],
        default=None,
    )

    return {
        "model": model,
        "runs": runs,
        "aggregates": aggregates,
        "worst_segment_risk": portfolio_worst,
        "watermark": watermark,
    }
