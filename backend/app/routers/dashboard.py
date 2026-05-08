"""GET /api/dashboard/summary — single-call aggregate for the dashboard."""

from __future__ import annotations

from fastapi import APIRouter

from app.routers.models import HOLDOUT_RESULTS_TABLE
from ml.model_registry import list_models, metrics_for
from services import donor_pool_service as dps
from services.persistence import read_table

router = APIRouter(prefix="/api/dashboard", tags=["dashboard"])


@router.get("/summary")
def summary() -> dict:
    """Aggregate state for the /dashboard hero cards and charts.

    Avoids 5 separate frontend calls by returning everything the dashboard
    needs in one response.
    """
    pool = dps.get_pool_size_status().to_dict()

    models = list_models()
    latest_model = (
        sorted(models, key=lambda r: r["created_at"], reverse=True)[0]
        if models
        else None
    )
    latest_metrics: list[dict] = []
    ablation: list[dict] = []
    weighted_r2: float | None = None
    bootstrap_ci: dict | None = None
    if latest_model:
        latest_metrics = metrics_for(latest_model["id"])
        for m in latest_metrics:
            if m["metric_type"] == "weighted_r_squared":
                weighted_r2 = m["value"]
            if m["metric_type"] == "weighted_r_squared_bootstrap_mean":
                bootstrap_ci = {
                    "mean": m["value"],
                    "ci_lower": m.get("ci_lower"),
                    "ci_upper": m.get("ci_upper"),
                }
        for m in latest_metrics:
            if m["metric_type"] == "ablation_weighted_r2":
                spec = (m.get("segment") or {}).get("spec")
                if spec:
                    ablation.append({"spec": spec, "weighted_r2": m["value"]})

    runs = read_table("prediction_runs")
    portfolio_runs = [r for r in runs if r.get("portfolio_run")]
    icpd_values = [r["predicted_icpd"] for r in portfolio_runs]
    portfolio_summary = None
    if icpd_values:
        risk_counts = {"severe": 0, "high": 0, "medium": 0, "low": 0, "unknown": 0}
        for r in portfolio_runs:
            risk = (r.get("worst_segment_risk") or {}).get("risk", "unknown")
            risk_counts[risk] = risk_counts.get(risk, 0) + 1
        portfolio_summary = {
            "n_runs": len(portfolio_runs),
            "mean_icpd": float(sum(icpd_values) / len(icpd_values)),
            "min_icpd": float(min(icpd_values)),
            "max_icpd": float(max(icpd_values)),
            "risk_counts": risk_counts,
            "icpd_values": icpd_values,
        }

    holdout_rows = read_table(HOLDOUT_RESULTS_TABLE)
    holdout_summary = None
    if holdout_rows:
        severity_counts = {"severe": 0, "high": 0, "medium": 0, "low": 0, "unknown": 0}
        for r in holdout_rows:
            severity_counts[r.get("risk", "unknown")] = (
                severity_counts.get(r.get("risk", "unknown"), 0) + 1
            )
        holdout_summary = {
            "n_levels": len(holdout_rows),
            "n_vars": len({r["segmentation_var"] for r in holdout_rows}),
            "severity_counts": severity_counts,
        }

    return {
        "donor_pool": pool,
        "latest_model": (
            {
                **latest_model,
                "weighted_r_squared": weighted_r2,
                "bootstrap": bootstrap_ci,
                "ablation": ablation,
            }
            if latest_model
            else None
        ),
        "portfolio": portfolio_summary,
        "holdouts": holdout_summary,
        "n_models_total": len(models),
        "n_prediction_runs_total": len(runs),
    }
