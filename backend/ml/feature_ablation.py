"""
Progressive feature ablation matching paper Figure 2 (V.4 Wave 2 — rebuilt on OOF).

Five model specifications:
  PIE(Pre)               pre-determined features only
  PIE(Pre+Yt)            adds outcome rate (conversions_per_dollar)
  PIE(Pre+Yt+LCC-7D)     adds LCC-7D-per-dollar
  PIE(Full)              all post-determined features
  Raw_LCC_7D_benchmark   predicts ICPD = LCC-7D / Cost (no model)

V.4 changes (paper §5.1):
  * Each spec is evaluated with **out-of-fold** predictions across a k-fold
    CV split (default k=10), not a 25% hold-out.
  * Each spec gets a **bootstrap CI** on weighted R² (n_draws=1000 default)
    so the chart can render error bars matching paper Figure 2.
  * Headline R² is consistent with the model-training-service definition:
    weighted R² on OOF predictions using true cost as the weight.

The Raw_LCC_7D_benchmark spec is non-parametric (no fit); we evaluate it on
the same observations the OOF predictions cover, so the comparison is fair.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from pie_formulas import bootstrap_metric, weighted_r_squared
from ml.train_random_forest import (
    DEFAULT_GRID,
    _build_pipeline,
    _flatten_feature_rows,
)
from services.feature_engineering_service import X_PRE_FIELDS

ABLATION_SPECS = [
    "PIE(Pre)",
    "PIE(Pre+Yt)",
    "PIE(Pre+Yt+LCC-7D)",
    "PIE(Full)",
    "Raw_LCC_7D_benchmark",
]


def _columns_for_spec(spec: str) -> list[str] | None:
    """Return the feature columns active in a given spec (None = all)."""
    if spec == "PIE(Pre)":
        return list(X_PRE_FIELDS)
    if spec == "PIE(Pre+Yt)":
        return list(X_PRE_FIELDS) + ["conversions_per_dollar"]
    if spec == "PIE(Pre+Yt+LCC-7D)":
        return list(X_PRE_FIELDS) + ["conversions_per_dollar", "lcc_7d_per_dollar"]
    if spec == "PIE(Full)":
        return None
    if spec == "Raw_LCC_7D_benchmark":
        return ["lcc_7d_per_dollar"]
    raise ValueError(f"unknown spec: {spec}")


@dataclass
class AblationRow:
    spec: str
    weighted_r2: float
    ci_lower: float
    ci_upper: float
    ci_mean: float
    n_observations: int
    n_splits: int
    n_bootstrap: int


def run_ablation(
    feature_rows: list[dict],
    label_rows: list[dict],
    cost_weights: list[float] | None = None,
    n_splits: int = 10,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> list[dict]:
    """V.4 paper-faithful ablation. Returns one dict per spec.

    Each dict has weighted_r2 + 95% bootstrap CI on the OOF prediction set.
    Smaller donor pools auto-reduce n_splits (clamped to ``len(common)-1``).
    """
    feats = _flatten_feature_rows(feature_rows).set_index("campaign_id")
    labels = pd.DataFrame(label_rows).set_index("campaign_id")["icpd"].astype(float)
    common = feats.index.intersection(labels.index)
    if len(common) < 4:
        raise ValueError(
            f"need at least 4 aligned (feature, label) rows; got {len(common)}"
        )
    X = feats.loc[common]
    y = labels.loc[common]
    w = (
        np.asarray(cost_weights, dtype=float)
        if cost_weights is not None
        else np.ones(len(common))
    )
    if len(w) != len(common):
        raise ValueError("cost_weights length must match aligned rows")

    effective_splits = max(2, min(n_splits, len(common) - 1))
    kf = KFold(n_splits=effective_splits, shuffle=True, random_state=seed)
    params = DEFAULT_GRID[0]  # ablation uses the cheap spec; full grid in train()

    out: list[AblationRow] = []
    y_arr = y.to_numpy()

    for spec in ABLATION_SPECS:
        if spec == "Raw_LCC_7D_benchmark":
            # Non-parametric benchmark — no fit, no CV. Evaluate against
            # observed ICPD using the same weighted-R² + bootstrap procedure
            # so it sits on the same axis as the model specs.
            if "lcc_7d_per_dollar" in X.columns:
                preds = X["lcc_7d_per_dollar"].astype(float).fillna(0.0).to_numpy()
            else:
                preds = np.zeros(len(common))
        else:
            cols = _columns_for_spec(spec)
            keep = [c for c in (cols or X.columns) if c in X.columns]
            X_use = X[keep]
            preds = np.full(len(common), np.nan, dtype=float)
            for tr_idx, te_idx in kf.split(X_use):
                pipe, _ = _build_pipeline(X_use.iloc[tr_idx], params)
                pipe.fit(X_use.iloc[tr_idx], y.iloc[tr_idx])
                preds[te_idx] = pipe.predict(X_use.iloc[te_idx])

        # Drop any NaN slots (shouldn't happen with KFold clamp, defensive).
        mask = ~np.isnan(preds)
        y_eff = y_arr[mask]
        p_eff = preds[mask]
        w_eff = w[mask]

        try:
            r2 = weighted_r_squared(y_eff.tolist(), p_eff.tolist(), w_eff.tolist())
        except ValueError:
            r2 = float("nan")

        try:
            boot = bootstrap_metric(
                lambda yt, yp, ww: weighted_r_squared(
                    yt.tolist(), yp.tolist(), ww.tolist()
                ),
                y_eff,
                p_eff,
                w_eff,
                n_draws=n_bootstrap,
            )
            ci_lower = float(boot["p025"])
            ci_upper = float(boot["p975"])
            ci_mean = float(boot["mean"])
        except (ValueError, ZeroDivisionError):
            ci_lower = float("nan")
            ci_upper = float("nan")
            ci_mean = float("nan")

        out.append(
            AblationRow(
                spec=spec,
                weighted_r2=r2,
                ci_lower=ci_lower,
                ci_upper=ci_upper,
                ci_mean=ci_mean,
                n_observations=int(mask.sum()),
                n_splits=effective_splits,
                n_bootstrap=n_bootstrap,
            )
        )

    return [asdict(r) for r in out]
