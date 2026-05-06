"""
Hold-out-one-level extrapolation test (PDF §5.3, Table 1; Prompt 2.4).

For each level ℓ in a segmentation variable:
  within_R²        = train on a random half of level-ℓ RCTs, score on the
                     held-out half (interpolation baseline)
  extrap_R²        = train on an equal-sized sample drawn from outside ℓ,
                     score on the held-out level-ℓ rows (extrapolation)
Repeat `n_iterations` times and report distributions + median penalty.

Targets: 6 segmentation variables (advertiser_size, campaign_year,
custom_audience, conversion_optimization, prospecting_vs_retargeting, vertical).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from pie_formulas import weighted_r_squared
from ml.train_random_forest import (
    DEFAULT_GRID,
    _build_pipeline,
    _flatten_feature_rows,
)

SEGMENTATION_VARS: tuple[str, ...] = (
    "vertical",
    "audience_type",
    "conversion_optimization",
    "custom_audience",
    "advertiser_platform_experience_months",
    "month",  # year-proxy on small synthetic donor pools; swap to year IRL
)

# Categorical risk bands — calibrated to PDF Table 1 baselines.
RISK_BANDS = (
    ("severe", 25.0),
    ("high", 15.0),
    ("medium", 5.0),
    ("low", 0.0),
)


@dataclass
class ExtrapolationLevelResult:
    segmentation_var: str
    level: str
    within_r2_median: float
    extrapolation_r2_median: float
    penalty_pp: float
    n_iterations: int
    risk: str

    def to_dict(self) -> dict:
        return asdict(self)


def _classify(penalty_pp: float) -> str:
    for label, threshold in RISK_BANDS:
        if penalty_pp >= threshold:
            return label
    return "low"


def _train_and_score(
    train_rows: pd.DataFrame, train_y: pd.Series, score_rows: pd.DataFrame, score_y: pd.Series, weights: np.ndarray
) -> float:
    if train_rows.empty or score_rows.empty:
        return float("nan")
    pipe, _ = _build_pipeline(train_rows, DEFAULT_GRID[0])
    pipe.fit(train_rows, train_y)
    preds = pipe.predict(score_rows)
    try:
        return weighted_r_squared(score_y.tolist(), preds.tolist(), weights.tolist())
    except ValueError:
        return float("nan")


def run_extrapolation_test(
    feature_rows: list[dict],
    label_rows: list[dict],
    segmentation_var: str,
    n_iterations: int = 50,
    seed: int = 42,
) -> list[dict]:
    """Run the hold-out-one-level test for every observed level of segmentation_var."""
    if segmentation_var not in SEGMENTATION_VARS:
        raise ValueError(
            f"segmentation_var must be one of {SEGMENTATION_VARS}, got {segmentation_var}"
        )
    feats = _flatten_feature_rows(feature_rows).set_index("campaign_id")
    if segmentation_var not in feats.columns:
        raise ValueError(
            f"segmentation_var '{segmentation_var}' not in feature columns"
        )
    labels = pd.DataFrame(label_rows).set_index("campaign_id")["icpd"].astype(float)
    common = feats.index.intersection(labels.index)
    if len(common) < 8:
        raise ValueError(
            f"need at least 8 aligned (feature, label) rows; got {len(common)}"
        )
    X = feats.loc[common].copy()
    y = labels.loc[common]

    levels = [str(l) for l in X[segmentation_var].dropna().unique()]
    rng = np.random.default_rng(seed)

    results: list[ExtrapolationLevelResult] = []
    for level in levels:
        within_scores: list[float] = []
        extrap_scores: list[float] = []
        for _ in range(n_iterations):
            in_level = X[X[segmentation_var].astype(str) == level]
            out_of_level = X[X[segmentation_var].astype(str) != level]
            if len(in_level) < 4 or len(out_of_level) < 2:
                continue
            half = len(in_level) // 2
            shuffled = in_level.sample(frac=1.0, random_state=int(rng.integers(0, 2**31)))
            train_in = shuffled.iloc[:half]
            test_in = shuffled.iloc[half:]
            extrap_n = min(len(out_of_level), max(half, 1))
            train_out = out_of_level.sample(
                n=extrap_n, random_state=int(rng.integers(0, 2**31)),
                replace=len(out_of_level) < extrap_n,
            )

            w_test = np.ones(len(test_in))
            within_scores.append(
                _train_and_score(
                    train_in, y.loc[train_in.index], test_in, y.loc[test_in.index], w_test
                )
            )
            extrap_scores.append(
                _train_and_score(
                    train_out, y.loc[train_out.index], test_in, y.loc[test_in.index], w_test
                )
            )

        within_med = float(np.nanmedian(within_scores)) if within_scores else float("nan")
        extrap_med = float(np.nanmedian(extrap_scores)) if extrap_scores else float("nan")
        penalty_pp = (
            (within_med - extrap_med) * 100.0
            if not (np.isnan(within_med) or np.isnan(extrap_med))
            else float("nan")
        )
        results.append(
            ExtrapolationLevelResult(
                segmentation_var=segmentation_var,
                level=str(level),
                within_r2_median=within_med,
                extrapolation_r2_median=extrap_med,
                penalty_pp=penalty_pp,
                n_iterations=len(within_scores),
                risk=_classify(penalty_pp) if not np.isnan(penalty_pp) else "unknown",
            )
        )
    return [r.to_dict() for r in results]
