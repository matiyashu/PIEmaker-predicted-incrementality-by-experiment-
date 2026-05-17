"""V.4 Wave 3 (Phase 5) — disagreement curves over 0.5×–1.5× segment median,
PIE vs Raw-LCC-7D side-by-side, and the PIEmaker-extension EDC curve."""

from __future__ import annotations

import random

import pytest

from pie_formulas.decision_curves import (
    disagreement_curves,
    disagreement_curves_compare,
    expected_disagreement_cost_curve,
)


def _make_pool(n: int = 60, seed: int = 7) -> tuple[list[float], list[float], list[float]]:
    """Synthetic ICPD + PIE-like prediction + Raw-LCC-like prediction.

    Design: PIE is a noisy unbiased estimator (well-calibrated). Raw-LCC is
    biased upward (typical of attribution-based baselines) so the PIE
    disagreement curve should sit below the Raw-LCC curve across most
    thresholds — that's the paper-faithful sanity check.
    """
    rng = random.Random(seed)
    icpd_true: list[float] = []
    pie_pred: list[float] = []
    lcc_pred: list[float] = []
    for _ in range(n):
        true = rng.uniform(0.02, 0.20)
        icpd_true.append(true)
        pie_pred.append(true + rng.gauss(0, 0.012))
        # Raw-LCC overstates: positive bias + heavier noise.
        lcc_pred.append(true * 1.35 + rng.gauss(0, 0.03))
    return icpd_true, pie_pred, lcc_pred


# --- disagreement_curves ------------------------------------------------------


def test_disagreement_curves_scans_default_range_inclusive():
    """Default 0.5×–1.5× in 0.05 steps → 21 threshold points."""
    icpd_true, pie_pred, _ = _make_pool(30)
    curve = disagreement_curves(icpd_true, pie_pred, reference_median=0.1)
    assert len(curve) == 21
    assert curve[0]["threshold_ratio"] == pytest.approx(0.5, abs=1e-9)
    assert curve[-1]["threshold_ratio"] == pytest.approx(1.5, abs=1e-9)
    for point in curve:
        assert 0.0 <= point["disagreement"] <= 1.0
        assert 0.0 <= point["type_1"] <= 1.0
        assert 0.0 <= point["type_2"] <= 1.0
        # Disagreement is exactly the sum of Type-I + Type-II (paper Eq.)
        assert point["disagreement"] == pytest.approx(
            point["type_1"] + point["type_2"], abs=1e-9
        )


def test_disagreement_curves_uses_median_when_reference_omitted():
    icpd_true = [0.1, 0.2, 0.3, 0.4, 0.5]
    pie_pred = [0.12, 0.21, 0.29, 0.41, 0.49]
    curve = disagreement_curves(icpd_true, pie_pred)
    # Median is 0.3 → threshold ratio 1.0 corresponds to t = 0.3.
    midpoint = next(p for p in curve if p["threshold_ratio"] == pytest.approx(1.0))
    assert midpoint["threshold"] == pytest.approx(0.3, abs=1e-9)


def test_disagreement_curves_rejects_mismatched_arrays():
    with pytest.raises(ValueError):
        disagreement_curves([0.1, 0.2], [0.1, 0.2, 0.3], reference_median=0.1)


def test_disagreement_curves_rejects_nonpositive_reference():
    with pytest.raises(ValueError):
        disagreement_curves([0.1, 0.2], [0.1, 0.2], reference_median=0.0)


# --- PIE vs Raw-LCC compare ---------------------------------------------------


def test_pie_beats_lcc_on_synthetic_pool_where_lcc_is_biased():
    icpd_true, pie_pred, lcc_pred = _make_pool(80, seed=11)
    result = disagreement_curves_compare(
        icpd_true, pie_pred, lcc_pred
    )
    pie_at_median = next(
        p for p in result["pie"] if p["threshold_ratio"] == pytest.approx(1.0)
    )
    lcc_at_median = next(
        p for p in result["raw_lcc"] if p["threshold_ratio"] == pytest.approx(1.0)
    )
    # Paper baselines: PIE 8–12% disagreement vs RCT, LCC 12–20%. On this
    # synthetic pool, PIE should disagree less than LCC at the segment median.
    assert pie_at_median["disagreement"] < lcc_at_median["disagreement"]


def test_compare_has_matching_threshold_ratios():
    icpd_true, pie_pred, lcc_pred = _make_pool(40, seed=13)
    result = disagreement_curves_compare(icpd_true, pie_pred, lcc_pred)
    pie_ratios = [p["threshold_ratio"] for p in result["pie"]]
    lcc_ratios = [p["threshold_ratio"] for p in result["raw_lcc"]]
    assert pie_ratios == lcc_ratios


# --- Expected disagreement cost (PIEmaker extension) --------------------------


def test_edc_curve_adds_expected_cost_and_extension_flag():
    icpd_true, pie_pred, _ = _make_pool(30, seed=17)
    curve = disagreement_curves(icpd_true, pie_pred, reference_median=0.1)
    edc = expected_disagreement_cost_curve(
        curve, fp_cost_per_unit=2.0, fn_cost_per_unit=1.5
    )
    assert len(edc) == len(curve)
    for raw, decorated in zip(curve, edc):
        expected = raw["type_1"] * 2.0 + raw["type_2"] * 1.5
        assert decorated["expected_cost"] == pytest.approx(expected, abs=1e-9)
        assert decorated["is_piemaker_extension"] is True


def test_edc_curve_rejects_negative_costs():
    curve = [{"threshold_ratio": 1.0, "threshold": 0.1, "disagreement": 0.1,
              "type_1": 0.05, "type_2": 0.05}]
    with pytest.raises(ValueError):
        expected_disagreement_cost_curve(curve, fp_cost_per_unit=-1.0, fn_cost_per_unit=1.0)
    with pytest.raises(ValueError):
        expected_disagreement_cost_curve(curve, fp_cost_per_unit=1.0, fn_cost_per_unit=-1.0)
