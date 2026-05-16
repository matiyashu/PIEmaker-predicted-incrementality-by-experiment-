"""
Model training orchestration (Prompt 2.3 + V.4 Wave 1).

Pulls feature rows + label rows from persistence, runs the cross-validated
training pipeline, computes the headline diagnostics (weighted R², bootstrap
CI, R² ceiling, feature ablation, LCC slope/ρ), captures the concept-drift
baseline, and registers the model in the file-based registry.

V.4 changes:
  * Headline weighted R² computed from OOF predictions (TrainingArtifacts
    .oof_predictions), not from a full-data refit.
  * Bootstrap also runs on OOF predictions, n_draws default = 1000.
  * Default n_splits = 10 (paper §5.1).
  * Sample weights = true campaign cost from feature_store.cost. Falls back
    to the V.3 (1/conversions_per_dollar) proxy only when cost is missing.
    The fallback fact is captured in the model card so the trust UI can
    surface a deviation badge.
  * R² ceiling uses pie_formulas.r_squared_ceiling_from_label_noise when
    per-RCT ATT standard errors are derivable; falls back to residual
    variance otherwise (also a model-card deviation).
"""

from __future__ import annotations

import math

import numpy as np

from pie_formulas import (
    bootstrap_metric,
    icpd_label_variance,
    lcc_ols_slope,
    lcc_spearman_rho,
    r_squared_ceiling,
    r_squared_ceiling_from_label_noise,
    weighted_r_squared,
)
from pie_formulas.model_card_schema import ModelCardCriteria, evaluate_alignment
from ml.feature_ablation import run_ablation
from ml.model_registry import record_metric, register_model
from ml.train_random_forest import train
from services.donor_pool_service import get_pool_size_status
from services.persistence import read_table


def _load_aligned_rows(
    feature_set_version: str,
) -> tuple[list[dict], list[dict]]:
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
    return feats, labels


def _cost_proxy_for(feat_row: dict) -> float:
    """V.3 fallback weight used only when feature_store.cost is missing.

    Reciprocal of `conversions_per_dollar` is a rough cost proxy and is
    intentionally suboptimal — paper requires true campaign cost.
    """
    cpd = (feat_row.get("x_post") or {}).get("conversions_per_dollar")
    try:
        v = float(cpd) if cpd is not None else 1.0
    except (TypeError, ValueError):
        v = 1.0
    return max(1.0, v) ** -1


def _resolve_weights(
    feats_aligned: list[dict],
) -> tuple[list[float], str]:
    """Resolve weights for paper-aligned R². Returns (weights, source)."""
    costs: list[float] = []
    missing = False
    for f in feats_aligned:
        c = f.get("cost")
        if c is None:
            missing = True
            costs.append(_cost_proxy_for(f))
        else:
            try:
                costs.append(float(c))
            except (TypeError, ValueError):
                missing = True
                costs.append(_cost_proxy_for(f))
    source = "true_cost" if not missing else "proxy"
    return costs, source


def _compute_r2_ceiling(
    oof_cids: list[str],
    icpd_oof: list[float],
    pred_oof: list[float],
    label_by_id: dict[str, dict],
    cost_by_id: dict[str, float],
) -> tuple[float, str]:
    """V.4 paper-faithful R² ceiling with graceful fallback.

    Prefers label-noise-based ceiling (paper §5.2) when per-RCT Sample-1
    counts are present in the label rows. Falls back to the legacy
    residual-variance ceiling only for legacy fixtures that pre-date the
    Wave 1 label-generator. The method used is returned alongside the
    ceiling value so the model card can record the deviation.
    """
    label_variances: list[float] = []
    cost_weights: list[float] = []
    can_use_label_noise = True
    for cid in oof_cids:
        label = label_by_id.get(cid) or {}
        s1_test_users = label.get("sample_1_test_users")
        s1_control_users = label.get("sample_1_control_users")
        s1_test_conv = label.get("sample_1_test_conversions")
        s1_control_conv = label.get("sample_1_control_conversions")
        exposure = label.get("exposure_rate")
        cost = label.get("cost") or cost_by_id.get(cid)
        if None in (
            s1_test_users,
            s1_control_users,
            s1_test_conv,
            s1_control_conv,
            exposure,
            cost,
        ):
            can_use_label_noise = False
            break
        try:
            var = icpd_label_variance(
                test_conversions=float(s1_test_conv),
                test_users=float(s1_test_users),
                control_conversions=float(s1_control_conv),
                control_users=float(s1_control_users),
                exposure_rate_value=float(exposure),
                cost=float(cost),
            )
        except ValueError:
            can_use_label_noise = False
            break
        label_variances.append(var)
        cost_weights.append(float(cost))

    if can_use_label_noise and label_variances:
        try:
            ceiling = r_squared_ceiling_from_label_noise(
                icpd_label_variances=label_variances,
                icpd_labels=icpd_oof,
                weights=cost_weights,
            )
            return ceiling, "label_noise"
        except ValueError:
            pass

    # Legacy fallback: residual-variance ceiling.
    ceiling = r_squared_ceiling(
        outcome_noise_variance=max(
            1e-9, float(np.var(np.array(icpd_oof) - np.array(pred_oof)))
        ),
        total_outcome_variance=max(1e-9, float(np.var(icpd_oof))),
    )
    return ceiling, "residual"


def train_pie_model(
    feature_set_version: str = "v1",
    name: str = "pie_random_forest",
    n_bootstrap: int = 1000,
    grid: list[dict] | None = None,
    n_splits: int = 10,
) -> dict:
    """V.4 paper-mode training orchestrator.

    Defaults are paper-aligned: 10-fold CV, 1000-bootstrap on OOF, true cost
    weights. Smaller pools may scale these down automatically (KFold clamps
    n_splits to len(common)-1) and the deviation is captured in the model
    card.
    """
    pool_status = get_pool_size_status()
    if pool_status.band == "blocked":
        raise PermissionError(
            f"Donor pool size {pool_status.n_admitted} below 200 RCTs; "
            "training blocked (PDF §4.2)."
        )
    feats, labels = _load_aligned_rows(feature_set_version)
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

    # V.4: true cost weights, with documented fallback to proxy.
    cost_weights, weights_source = _resolve_weights(feats_aligned)
    feat_by_id = {f["campaign_id"]: f for f in feats_aligned}
    cost_by_id = dict(zip([f["campaign_id"] for f in feats_aligned], cost_weights))

    # Train with OOF return enabled by default.
    artifacts = train(
        feats_aligned,
        list(label_by_id.values()),
        weights=cost_weights,
        grid=grid,
        n_splits=n_splits,
        return_oof=True,
    )

    # V.4: headline R² + bootstrap come from OOF, not full-data refit.
    # `artifacts.oof_campaign_ids` preserves the canonical order matching
    # `artifacts.oof_predictions`; align labels and weights to it.
    oof_cids = artifacts.oof_campaign_ids or [
        f["campaign_id"] for f in feats_aligned
    ]
    oof_preds_raw = artifacts.oof_predictions or []

    icpd_oof: list[float] = []
    pred_oof: list[float] = []
    w_oof: list[float] = []
    for cid, p in zip(oof_cids, oof_preds_raw):
        # KFold may leave NaNs only if k > n; we already clamp inside train()
        # so every row is covered. Defensive guard anyway:
        if p is None or (isinstance(p, float) and math.isnan(p)):
            continue
        icpd_oof.append(float(label_by_id[cid]["icpd"]))
        pred_oof.append(float(p))
        w_oof.append(float(cost_by_id.get(cid, 1.0)))

    r2_oof = weighted_r_squared(icpd_oof, pred_oof, w_oof)
    boot = bootstrap_metric(
        lambda yt, yp, w: weighted_r_squared(yt.tolist(), yp.tolist(), w.tolist()),
        np.array(icpd_oof),
        np.array(pred_oof),
        np.array(w_oof),
        n_draws=n_bootstrap,
    )

    # V.4 R² ceiling: label-noise-based when per-RCT Sample-1 counts are
    # available (Phase 2c). Falls back to residual-variance ceiling only if
    # the label rows are missing the per-arm counts (legacy fixtures).
    ceiling, ceiling_method = _compute_r2_ceiling(
        oof_cids=oof_cids,
        icpd_oof=icpd_oof,
        pred_oof=pred_oof,
        label_by_id=label_by_id,
        cost_by_id=cost_by_id,
    )

    # Optional LCC diagnostics — requires lcc_7d_per_dollar in features.
    # Use OOF order so the diagnostic is consistent with the headline R².
    lcc_slope = None
    lcc_rho = None
    lcc_per_dollar = [
        (feat_by_id[cid].get("x_post") or {}).get("lcc_7d_per_dollar")
        for cid in oof_cids
    ]
    if all(v is not None for v in lcc_per_dollar) and len(lcc_per_dollar) >= 2:
        try:
            lcc_slope = lcc_ols_slope(icpd_oof, lcc_per_dollar)
            lcc_rho = lcc_spearman_rho(icpd_oof, lcc_per_dollar)
        except ValueError:
            pass

    # Ablation still uses the V.3 implementation (25% hold-out) — Wave 2
    # Phase 3 rebuilds it on OOF with bootstrap error bars.
    ablation = run_ablation(feats_aligned, list(label_by_id.values()), cost_weights)

    # Concept-drift baseline: feature importances per fitted RF.
    rf = artifacts.estimator.named_steps["rf"]
    importances = rf.feature_importances_.tolist() if hasattr(rf, "feature_importances_") else []

    # Build V.4 model card so the trust UI can surface deviations.
    criteria = ModelCardCriteria(
        eval_mode="paper",
        n_splits=int(artifacts.oof_n_splits or n_splits),
        n_bootstrap=n_bootstrap,
        weights_source=weights_source,  # "true_cost" or "proxy"
        headline_r2_basis="oof",
        r2_ceiling_method=ceiling_method,  # "residual" until Phase 2c
    )
    paper_aligned, deviations = evaluate_alignment(criteria)
    model_card_payload = {
        "paper_aligned": paper_aligned,
        "deviations": deviations,
        "criteria": criteria.to_dict(),
        "paper_alignment_version": "v4.0.0",
    }

    model = register_model(
        name=name,
        algorithm="random_forest",
        feature_set_version=feature_set_version,
        hyperparameters=artifacts.chosen_hyperparameters,
        training_donor_pool_size=pool_status.n_admitted,
        estimator=artifacts.estimator,
        concept_drift_baseline={
            "feature_importances": importances,
            "model_card": model_card_payload,
        },
        status="research" if pool_status.band == "research_mode" else "production",
    )

    record_metric(model["id"], "weighted_r_squared", r2_oof)
    record_metric(model["id"], "weighted_r_squared_oof", r2_oof)
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
        "weighted_r_squared": r2_oof,
        "bootstrap": boot,
        "r_squared_ceiling": ceiling,
        "lcc_diagnostics": {"slope": lcc_slope, "spearman_rho": lcc_rho},
        "ablation": ablation,
        "donor_pool_status": pool_status.to_dict(),
        # V.4 NEW
        "model_card": model_card_payload,
        "oof": {
            "n_splits": artifacts.oof_n_splits,
            "n_predictions": len(pred_oof),
        },
    }
