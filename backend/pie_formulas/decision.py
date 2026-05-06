"""
Decision metrics (PDF Section 3.4).

Converts model output into go/no-go business decisions: segment-relative
thresholds, disagreement probability with Type I/II decomposition, and the new
v3 Expected Disagreement Cost (EDC) view.
"""

from __future__ import annotations

from typing import Sequence, Union

import numpy as np

ThresholdLike = Union[float, Sequence[float], np.ndarray]


def segment_median_threshold(
    icpd_values: Sequence[float],
) -> float:
    """Segment median threshold ICPD*_seg = median(ICPD_r) within segment.

    Reference: Gordon et al. 2026, p. 31.
    Callers must group rows by (vertical, funnel) before passing in; this
    function operates on a single segment's ICPD values.
    """
    arr = np.asarray(icpd_values, dtype=float)
    if arr.size == 0:
        raise ValueError("icpd_values must be non-empty")
    return float(np.median(arr))


def threshold_scan_range(
    segment_median: float,
    low_multiplier: float = 0.5,
    high_multiplier: float = 1.5,
    step: float = 0.05,
) -> list[float]:
    """Default scan range for the Decision Simulator.

    ICPD* ∈ [low_multiplier, high_multiplier] × ICPD*_seg, stepped by `step`.

    Reference: Gordon et al. 2026, p. 31.
    Default scan: 50% → 150% of segment median in 5% steps.
    """
    if low_multiplier <= 0:
        raise ValueError("low_multiplier must be > 0")
    if high_multiplier < low_multiplier:
        raise ValueError("high_multiplier must be >= low_multiplier")
    if step <= 0:
        raise ValueError("step must be > 0")
    n_steps = int(round((high_multiplier - low_multiplier) / step)) + 1
    return [
        segment_median * (low_multiplier + i * step) for i in range(n_steps)
    ]


def _to_threshold_array(
    threshold: ThresholdLike, n: int
) -> np.ndarray:
    """Broadcast a scalar or per-row threshold to length n."""
    if np.isscalar(threshold):
        return np.full(n, float(threshold))
    arr = np.asarray(threshold, dtype=float)
    if arr.shape[0] != n:
        raise ValueError(
            f"per-row threshold length {arr.shape[0]} does not match n={n}"
        )
    return arr


def disagreement_probability(
    icpd_true: Sequence[float],
    icpd_pred: Sequence[float],
    threshold: ThresholdLike,
) -> dict:
    """Disagreement probability with Type I / Type II decomposition.

    D(ICPD*) = (1/R) Σ [Type_I + Type_II]
        Type_I = 1{ICPD_r ≤ ICPD*} × 1{ICPD̂_r > ICPD*}   (false positive)
        Type_II = 1{ICPD_r > ICPD*} × 1{ICPD̂_r ≤ ICPD*}   (false negative)

    Reference: Section 6.1, Gordon et al. 2026, p. 31.
    Paper baselines: PIE 8–12% disagreement vs RCT, LCC 12–20%.

    `threshold` may be a scalar (absolute) or a per-row array (segment-relative,
    each row's segment-median × multiplier).
    """
    y_true = np.asarray(icpd_true, dtype=float)
    y_pred = np.asarray(icpd_pred, dtype=float)
    if y_true.shape != y_pred.shape:
        raise ValueError("icpd_true and icpd_pred must have the same shape")
    if y_true.size == 0:
        raise ValueError("inputs must be non-empty")
    thr = _to_threshold_array(threshold, y_true.shape[0])
    type_1 = ((y_true <= thr) & (y_pred > thr)).astype(int)
    type_2 = ((y_true > thr) & (y_pred <= thr)).astype(int)
    return {
        "disagreement_probability": float((type_1 + type_2).mean()),
        "type_1_error": float(type_1.mean()),
        "type_2_error": float(type_2.mean()),
    }


def expected_disagreement_cost(
    icpd_true: Sequence[float],
    icpd_pred: Sequence[float],
    threshold: ThresholdLike,
    campaign_cost: Sequence[float],
    forgone_inc_revenue: Sequence[float],
    cost_per_fp: float = 1.0,
    cost_per_fn: float = 1.0,
) -> float:
    """Expected dollar cost of model-vs-RCT disagreement (NEW in v3).

    EDC = P(Type_I) × Cost_FP × Campaign_Cost
        + P(Type_II) × Cost_FN × Forgone_Inc_Revenue

    Reference: Validation Memo §5; extends Gordon et al. 2026, §6 (p. 33),
    where the authors note asymmetric error costs but do not quantify them.

    Cost_FP interpretation: per-dollar cost of campaign budget when the model
    says "scale" but truth is "fail" (wasted spend).
    Cost_FN interpretation: per-dollar cost of forgone incremental revenue
    when the model says "pause" but truth is "success".
    """
    if cost_per_fp < 0 or cost_per_fn < 0:
        raise ValueError("cost_per_fp and cost_per_fn must be >= 0")
    y_true = np.asarray(icpd_true, dtype=float)
    y_pred = np.asarray(icpd_pred, dtype=float)
    cost = np.asarray(campaign_cost, dtype=float)
    forgone = np.asarray(forgone_inc_revenue, dtype=float)
    if not (y_true.shape == y_pred.shape == cost.shape == forgone.shape):
        raise ValueError("all input arrays must have the same shape")
    if y_true.size == 0:
        raise ValueError("inputs must be non-empty")
    thr = _to_threshold_array(threshold, y_true.shape[0])
    type_1 = (y_true <= thr) & (y_pred > thr)
    type_2 = (y_true > thr) & (y_pred <= thr)
    fp_cost = (type_1.astype(float) * cost_per_fp * cost).sum()
    fn_cost = (type_2.astype(float) * cost_per_fn * forgone).sum()
    return float(fp_cost + fn_cost)
