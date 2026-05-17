"""
Decision-disagreement curves (V.4 Wave 3 — Phase 5, paper §6.3 Fig 6 right).

A *single* disagreement number at one threshold doesn't show how the model's
decisions behave around the segment's typical ICPD. The paper plots the
**curve** D(t) over thresholds t = ratio × segment_median for ratios in
[0.5, 1.5] — that's the chart this module produces.

Composes the existing pie_formulas.decision primitives:
  * `segment_median_threshold`  — ICPD*_seg = median of segment ICPDs
  * `threshold_scan_range`      — list of t values over the ratio sweep
  * `disagreement_probability`  — Type I/II decomposition at one t

This module adds:
  * `disagreement_curves(...)`         — one curve, one model
  * `disagreement_curves_compare(...)` — PIE vs Raw-LCC-7D side by side
  * `expected_disagreement_cost_curve(...)` — PIEmaker extension, clearly
    labelled as such (paper §6 notes asymmetric error costs without
    quantifying; we expose a knob for advertisers to plug their own costs).
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from pie_formulas.decision import (
    disagreement_probability,
    segment_median_threshold,
    threshold_scan_range,
)


def disagreement_curves(
    icpd_true: Sequence[float],
    icpd_pred: Sequence[float],
    reference_median: float | None = None,
    low_ratio: float = 0.5,
    high_ratio: float = 1.5,
    step: float = 0.05,
) -> list[dict]:
    """One disagreement curve scanned over [low_ratio, high_ratio] × median.

    `reference_median` is the segment median ICPD; if omitted, the median of
    `icpd_true` is used so the curve still renders for a single-segment pool.

    Returns one dict per scanned threshold:
        {
            "threshold_ratio": float,   # e.g. 0.95 for 0.95 × median
            "threshold":       float,   # ratio × reference_median
            "disagreement":    float,   # D(t)
            "type_1":          float,   # FP component
            "type_2":          float,   # FN component
        }
    """
    y_true = np.asarray(icpd_true, dtype=float)
    y_pred = np.asarray(icpd_pred, dtype=float)
    if y_true.shape != y_pred.shape:
        raise ValueError("icpd_true and icpd_pred must have the same shape")
    if y_true.size == 0:
        raise ValueError("inputs must be non-empty")

    ref = (
        float(reference_median)
        if reference_median is not None
        else segment_median_threshold(y_true.tolist())
    )
    if ref <= 0:
        raise ValueError("reference_median must be > 0")

    thresholds = threshold_scan_range(ref, low_ratio, high_ratio, step)
    out: list[dict] = []
    for t in thresholds:
        result = disagreement_probability(
            icpd_true=y_true.tolist(),
            icpd_pred=y_pred.tolist(),
            threshold=t,
        )
        out.append(
            {
                "threshold_ratio": round(t / ref, 6),
                "threshold": float(t),
                "disagreement": result["disagreement_probability"],
                "type_1": result["type_1_error"],
                "type_2": result["type_2_error"],
            }
        )
    return out


def disagreement_curves_compare(
    icpd_true: Sequence[float],
    pie_pred: Sequence[float],
    raw_lcc_pred: Sequence[float],
    reference_median: float | None = None,
    low_ratio: float = 0.5,
    high_ratio: float = 1.5,
    step: float = 0.05,
) -> dict:
    """Side-by-side disagreement curves for PIE vs Raw-LCC-7D.

    The paper-target output (Fig 6 right): two curves on one axis so you can
    read off the threshold ratio where PIE's disagreement crosses below the
    LCC benchmark. Returns both curves plus the reference_median used.
    """
    y_true = np.asarray(icpd_true, dtype=float)
    if reference_median is None:
        reference_median = segment_median_threshold(y_true.tolist())
    return {
        "reference_median": float(reference_median),
        "low_ratio": low_ratio,
        "high_ratio": high_ratio,
        "step": step,
        "pie": disagreement_curves(
            icpd_true=icpd_true,
            icpd_pred=pie_pred,
            reference_median=reference_median,
            low_ratio=low_ratio,
            high_ratio=high_ratio,
            step=step,
        ),
        "raw_lcc": disagreement_curves(
            icpd_true=icpd_true,
            icpd_pred=raw_lcc_pred,
            reference_median=reference_median,
            low_ratio=low_ratio,
            high_ratio=high_ratio,
            step=step,
        ),
    }


def expected_disagreement_cost_curve(
    curve: list[dict],
    fp_cost_per_unit: float = 1.0,
    fn_cost_per_unit: float = 1.0,
) -> list[dict]:
    """PIEmaker extension: scale each curve point by asymmetric error costs.

    Paper §6 notes the asymmetry (a false-positive scale-up wastes spend; a
    false-negative pause forgoes incremental revenue) but doesn't quantify
    it. This helper takes per-unit cost knobs the advertiser controls and
    returns the same curve with an added `expected_cost` field.

    Labelled in the docstring + JSON so it's never confused for a paper-
    faithful metric — see the V.4 build plan, Phase 5.
    """
    if fp_cost_per_unit < 0 or fn_cost_per_unit < 0:
        raise ValueError("fp_cost_per_unit and fn_cost_per_unit must be >= 0")
    out: list[dict] = []
    for point in curve:
        cost = (
            point["type_1"] * fp_cost_per_unit
            + point["type_2"] * fn_cost_per_unit
        )
        out.append(
            {
                **point,
                "expected_cost": float(cost),
                "is_piemaker_extension": True,
            }
        )
    return out
