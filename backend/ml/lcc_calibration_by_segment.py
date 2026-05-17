"""
LCC calibration by segment (V.4 Wave 2 — Phase 3, paper §5.4).

For each level of a segmentation variable, compute the calibration
diagnostics paper §5.4 calls for:
  * bias_ratio    — mean(LCC) / mean(ICPD); >1 means LCC overstates lift
  * ols_slope     — OLS regression slope of ICPD on LCC-per-dollar
  * spearman_rho  — rank-correlation between LCC ordering and ICPD ordering
  * raw_lcc_r2    — weighted R² using raw LCC-per-dollar as the prediction
  * residual_mean / residual_p10 / residual_p90 — distribution of
    (ICPD − LCC-per-dollar) within the segment

These quantify *where* the LCC benchmark fails (e.g., overstates lift in
high-funnel ecommerce campaigns) — a paper-faithful diagnostic that drives
the "is this segment worth a shadow RCT?" decision.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from pie_formulas import (
    lcc_bias_ratio,
    lcc_ols_slope,
    lcc_spearman_rho,
    weighted_r_squared,
)
from ml.train_random_forest import _flatten_feature_rows


@dataclass
class CalibrationRow:
    segmentation_var: str
    level: str
    n: int
    bias_ratio: float | None
    ols_slope: float | None
    spearman_rho: float | None
    raw_lcc_r2: float | None
    residual_mean: float
    residual_p10: float
    residual_p90: float


def calibration_by_segment(
    feature_rows: list[dict],
    label_rows: list[dict],
    segmentation_var: str,
    cost_weights: list[float] | None = None,
    min_n_per_level: int = 5,
) -> list[dict]:
    """Per-level LCC calibration diagnostics.

    `segmentation_var` is a column name from X_pre (e.g. "vertical",
    "audience_type"). Levels with fewer than `min_n_per_level` rows are
    skipped — the diagnostics are too noisy to be useful below that.
    """
    feats = _flatten_feature_rows(feature_rows).set_index("campaign_id")
    labels = pd.DataFrame(label_rows).set_index("campaign_id")["icpd"].astype(float)
    common = list(feats.index.intersection(labels.index))
    if not common:
        raise ValueError("no overlap between feature rows and label rows")

    if segmentation_var not in feats.columns:
        raise ValueError(
            f"segmentation_var={segmentation_var!r} not present in feature rows; "
            f"available columns: {sorted(feats.columns)[:10]}…"
        )
    if "lcc_7d_per_dollar" not in feats.columns:
        raise ValueError(
            "lcc_7d_per_dollar not present — required for LCC calibration"
        )

    X = feats.loc[common]
    y = labels.loc[common]
    weights = (
        np.asarray(cost_weights, dtype=float)
        if cost_weights is not None
        else np.ones(len(common))
    )
    if cost_weights is not None and len(cost_weights) != len(common):
        raise ValueError("cost_weights length must match aligned rows")

    rows: list[CalibrationRow] = []
    for level, idx in X.groupby(segmentation_var).groups.items():
        idx_list = list(idx)
        if len(idx_list) < min_n_per_level:
            continue
        y_lvl = y.loc[idx_list].to_numpy()
        lcc_lvl = X.loc[idx_list, "lcc_7d_per_dollar"].astype(float).fillna(0.0).to_numpy()
        # Map campaign_ids to positional indices in X so weights line up.
        pos = [common.index(cid) for cid in idx_list]
        w_lvl = weights[pos]

        try:
            bias = lcc_bias_ratio(lcc_lvl.tolist(), y_lvl.tolist())
        except ValueError:
            bias = None
        try:
            slope = lcc_ols_slope(y_lvl.tolist(), lcc_lvl.tolist())
        except ValueError:
            slope = None
        try:
            rho = lcc_spearman_rho(y_lvl.tolist(), lcc_lvl.tolist())
        except ValueError:
            rho = None
        try:
            raw_r2 = weighted_r_squared(y_lvl.tolist(), lcc_lvl.tolist(), w_lvl.tolist())
        except ValueError:
            raw_r2 = None

        residuals = y_lvl - lcc_lvl
        rows.append(
            CalibrationRow(
                segmentation_var=segmentation_var,
                level=str(level),
                n=len(idx_list),
                bias_ratio=bias,
                ols_slope=slope,
                spearman_rho=rho,
                raw_lcc_r2=raw_r2,
                residual_mean=float(np.mean(residuals)),
                residual_p10=float(np.percentile(residuals, 10)),
                residual_p90=float(np.percentile(residuals, 90)),
            )
        )

    rows.sort(key=lambda r: r.n, reverse=True)
    return [asdict(r) for r in rows]
