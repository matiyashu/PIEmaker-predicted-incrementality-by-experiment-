"""
Model training orchestration (Prompt 2.3).

Pulls feature rows + label rows from persistence, runs the cross-validated
training pipeline, computes the headline diagnostics (weighted R², bootstrap
CI, R² ceiling, feature ablation, LCC slope/ρ), captures the concept-drift
baseline, and registers the model in the file-based registry.
"""

from __future__ import annotations

import numpy as np

from pie_formulas import (
    bootstrap_metric,
    lcc_ols_slope,
    lcc_spearman_rho,
    r_squared_ceiling,
    weighted_r_squared,
)
from ml.feature_ablation import run_ablation
from ml.model_registry import record_metric, register_model
from ml.train_random_forest import (
    _flatten_feature_rows,
    predict_icpd,
    train,
)
from services.donor_pool_service import get_pool_size_status
from services.persistence import read_table


def _load_aligned_rows(feature_set_version: str) -> tuple[list[dict], list[dict], list[float]]:
    feats = [
        r
        for r in read_table("feature_store")
        if r.get("feature_set_version") == feature_set_version and r.get("mode") == "training"
    ]
    # Use every label row that has both a campaign_id and an icpd value; the
    # donor-pool admission decision is tracked separately in
    # donor_pool_membership and shouldn't gate which labels train sees.
    labels = [
        r
        for r in read_table("rct_labels")
        if r.get("campaign_id") and r.get("icpd") is not None
    ]
    return feats, labels, [1.0] * len(feats)


def train_pie_model(
    feature_set_version: str = "v1",
    name: str = "pie_random_forest",
    n_bootstrap: int = 200,
    grid: list[dict] | None = None,
    n_splits: int = 5,
) -> dict:
    pool_status = get_pool_size_status()
    if pool_status.band == "blocked":
        raise PermissionError(
            f"Donor pool size {pool_status.n_admitted} below 200 RCTs; "
            "training blocked (PDF §4.2)."
        )
    feats, labels, _ = _load_aligned_rows(feature_set_version)
    if not feats or not labels:
        raise ValueError(
            "no feature_store / rct_labels rows persisted; run /api/features/build "
            "and /api/labels/generate first."
        )

    label_by_id = {r["campaign_id"]: r for r in labels}
    feats_aligned = [f for f in feats if f["campaign_id"] in label_by_id]
    if not feats_aligned:
        feat_sample = sorted({f.get("campaign_id") for f in feats})[:3]
        label_sample = sorted(label_by_id.keys())[:3]
        raise ValueError(
            f"no overlap between feature rows and label rows; "
            f"{len(feats)} feature rows (sample ids: {feat_sample}), "
            f"{len(labels)} label rows (sample ids: {label_sample})"
        )

    icpd_true = [label_by_id[f["campaign_id"]]["icpd"] for f in feats_aligned]
    cost_proxy = [
        max(1.0, float(f.get("x_post", {}).get("conversions_per_dollar") or 1.0)) ** -1
        for f in feats_aligned
    ]  # placeholder weight if cost is not in feature_store

    artifacts = train(
        feats_aligned, list(label_by_id.values()), grid=grid, n_splits=n_splits
    )
    preds = predict_icpd(artifacts.estimator, feats_aligned).tolist()

    r2 = weighted_r_squared(icpd_true, preds, cost_proxy)
    boot = bootstrap_metric(
        lambda yt, yp, w: weighted_r_squared(yt.tolist(), yp.tolist(), w.tolist()),
        np.array(icpd_true),
        np.array(preds),
        np.array(cost_proxy),
        n_draws=n_bootstrap,
    )
    ceiling = r_squared_ceiling(
        outcome_noise_variance=max(1e-9, np.var(np.array(icpd_true) - np.array(preds))),
        total_outcome_variance=max(1e-9, float(np.var(icpd_true))),
    )

    # Optional LCC diagnostics — requires lcc_7d_per_dollar in features
    lcc_slope = None
    lcc_rho = None
    lcc_per_dollar = [
        f.get("x_post", {}).get("lcc_7d_per_dollar") for f in feats_aligned
    ]
    if all(v is not None for v in lcc_per_dollar) and len(lcc_per_dollar) >= 2:
        try:
            lcc_slope = lcc_ols_slope(icpd_true, lcc_per_dollar)
            lcc_rho = lcc_spearman_rho(icpd_true, lcc_per_dollar)
        except ValueError:
            pass

    ablation = run_ablation(feats_aligned, list(label_by_id.values()), cost_proxy)

    # Concept-drift baseline: feature importances per fitted RF.
    rf = artifacts.estimator.named_steps["rf"]
    importances = rf.feature_importances_.tolist() if hasattr(rf, "feature_importances_") else []

    model = register_model(
        name=name,
        algorithm="random_forest",
        feature_set_version=feature_set_version,
        hyperparameters=artifacts.chosen_hyperparameters,
        training_donor_pool_size=pool_status.n_admitted,
        estimator=artifacts.estimator,
        concept_drift_baseline={"feature_importances": importances},
        status="research" if pool_status.band == "research_mode" else "production",
    )

    record_metric(model["id"], "weighted_r_squared", r2)
    record_metric(
        model["id"],
        "weighted_r_squared_bootstrap_mean",
        boot["mean"],
        ci_lower=boot["p025"],
        ci_upper=boot["p975"],
    )
    record_metric(model["id"], "r_squared_ceiling", ceiling)
    if lcc_slope is not None:
        record_metric(model["id"], "lcc_ols_slope", lcc_slope)
    if lcc_rho is not None:
        record_metric(model["id"], "lcc_spearman_rho", lcc_rho)
    for row in ablation:
        record_metric(
            model["id"],
            "ablation_weighted_r2",
            row["weighted_r2"],
            segment={"spec": row["spec"]},
        )

    return {
        "model": model,
        "n_observations": artifacts.n_observations,
        "weighted_r_squared": r2,
        "bootstrap": boot,
        "r_squared_ceiling": ceiling,
        "lcc_diagnostics": {"slope": lcc_slope, "spearman_rho": lcc_rho},
        "ablation": ablation,
        "donor_pool_status": pool_status.to_dict(),
    }
