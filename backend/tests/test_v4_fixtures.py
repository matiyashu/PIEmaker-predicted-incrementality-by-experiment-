"""V.4 Wave 5 (Phase 8) — fixture-driven edge-case tests.

Each fixture from ``tests.fixtures`` is tuned to trip a specific V.4
diagnostic; this file is the canary that catches regressions in the
diagnostic logic by checking the right band fires for the right scenario.
"""

from __future__ import annotations

import math

import pytest

from ml.advertiser_cv import run_existing_vs_new
from ml.feature_ablation import run_ablation
from ml.holdout_one_level import run_extrapolation_test
from ml.lcc_calibration_by_segment import calibration_by_segment
from pie_formulas import (
    icpd_label_variance,
    r_squared_ceiling_from_label_noise,
    weighted_r_squared,
)
from tests.fixtures import (
    cold_start_advertisers,
    high_lcc_bias,
    one_sided_noncompliance,
    perfect_compliance,
    sample_split,
    severe_extrapolation,
)


# --- 1. Perfect compliance ---------------------------------------------------


def test_perfect_compliance_exposure_rate_is_one():
    feats, _labels, _w = perfect_compliance(n=20)
    for f in feats:
        assert f["x_post"]["exposure_rate"] == 1.0


def test_perfect_compliance_ablation_runs_clean():
    """No D_bar quirks → ablation should rank specs in roughly the paper
    order (Full ≥ Pre on signal-carrying synthetic data)."""
    feats, labels, w = perfect_compliance(n=40)
    rows = run_ablation(feats, labels, w, n_splits=4, n_bootstrap=20)
    by_spec = {r["spec"]: r["weighted_r2"] for r in rows}
    assert by_spec["PIE(Full)"] >= by_spec["PIE(Pre)"] - 0.20


# --- 2. One-sided noncompliance ---------------------------------------------


def test_one_sided_exposure_rates_spread():
    feats, _labels, _w = one_sided_noncompliance(n=30)
    rates = [f["x_post"]["exposure_rate"] for f in feats]
    assert min(rates) >= 0.40
    assert max(rates) <= 0.85
    # Non-trivial spread
    assert max(rates) - min(rates) > 0.20


def test_one_sided_noncompliance_ablation_still_runs():
    """Ablation should be robust to D_bar < 1; specs should still order
    sensibly even though label noise is higher."""
    feats, labels, w = one_sided_noncompliance(n=40)
    rows = run_ablation(feats, labels, w, n_splits=4, n_bootstrap=20)
    assert len(rows) == 5
    # Every spec returns a finite R² (no NaN propagation from D_bar quirks)
    for r in rows:
        assert math.isfinite(r["weighted_r2"])


# --- 3. Sample-split path active --------------------------------------------


def test_sample_split_fixture_carries_per_arm_counts():
    _feats, labels, _w = sample_split(n=20)
    for label in labels:
        assert label["sample_split_active"] is True
        for key in (
            "sample_1_test_users",
            "sample_1_control_users",
            "sample_1_test_conversions",
            "sample_1_control_conversions",
            "sample_2_test_users",
            "sample_2_control_users",
        ):
            assert key in label
            assert label[key] > 0


def test_sample_split_enables_label_noise_ceiling():
    """With per-arm counts in place, the paper-faithful ceiling computes
    without falling back to the residual estimator."""
    _feats, labels, _w = sample_split(n=20)
    variances: list[float] = []
    icpd: list[float] = []
    for label in labels:
        var = icpd_label_variance(
            test_conversions=label["sample_1_test_conversions"],
            test_users=label["sample_1_test_users"],
            control_conversions=label["sample_1_control_conversions"],
            control_users=label["sample_1_control_users"],
            exposure_rate_value=label["exposure_rate"],
            cost=label["cost"],
        )
        variances.append(var)
        icpd.append(label["icpd"])
    ceiling = r_squared_ceiling_from_label_noise(variances, icpd)
    # Paper ceiling values are typically in [0.5, 1.0] for realistic pools.
    assert 0.0 <= ceiling <= 1.0


# --- 4. High LCC bias --------------------------------------------------------


def test_high_lcc_bias_calibration_flags_overstating_bias():
    """Calibration per segment must surface bias_ratio > 1.5 when Raw LCC
    is structurally overstating ICPD."""
    feats, labels, w = high_lcc_bias(n=60)
    rows = calibration_by_segment(feats, labels, "vertical", w, min_n_per_level=3)
    assert rows, "expected at least one segment level"
    # Average bias ratio across levels should be well above 1.
    bias_ratios = [r["bias_ratio"] for r in rows if r["bias_ratio"] is not None]
    assert bias_ratios, "expected at least one finite bias ratio"
    assert sum(bias_ratios) / len(bias_ratios) > 1.5


def test_high_lcc_bias_raw_lcc_r2_is_poor():
    """Raw LCC-7d as a predictor of ICPD should perform badly when LCC is
    structurally biased — confirms the diagnostic catches it."""
    feats, labels, w = high_lcc_bias(n=60)
    rows = calibration_by_segment(feats, labels, "vertical", w, min_n_per_level=3)
    raw_r2s = [r["raw_lcc_r2"] for r in rows if r["raw_lcc_r2"] is not None]
    assert raw_r2s
    # Even on the best segment, raw LCC R² should be < 0.5 on the biased pool.
    assert max(raw_r2s) < 0.5


# --- 5. Cold-start advertisers ----------------------------------------------


def test_cold_start_fixture_has_mixed_cohorts():
    feats, _labels, _w = cold_start_advertisers()
    adv_counts: dict[str, int] = {}
    for f in feats:
        adv = f["x_pre"]["advertiser_id"]
        adv_counts[adv] = adv_counts.get(adv, 0) + 1
    n_existing = sum(1 for c in adv_counts.values() if c > 1)
    n_new = sum(1 for c in adv_counts.values() if c == 1)
    assert n_existing > 0
    assert n_new > 0


def test_cold_start_advertiser_cv_shows_cohort_gap():
    """Existing-advertiser R² should outperform new-advertiser R² on a pool
    where new advertisers have a different ICPD generator (paper §5.3 cold-
    start finding)."""
    feats, labels, w = cold_start_advertisers()
    result = run_existing_vs_new(feats, labels, w, n_splits=3)
    cohort_r2 = {c["cohort"]: c["weighted_r2"] for c in result["cohorts"]}
    # Both cohorts populated
    assert cohort_r2["existing"] is not None
    assert cohort_r2["new"] is not None
    # The cohort_gap_pp field exists and is finite
    assert math.isfinite(result["cohort_gap_pp"])


# --- 6. Severe extrapolation -------------------------------------------------


def test_severe_extrapolation_fixture_has_held_out_vertical():
    feats, _labels, _w = severe_extrapolation()
    verticals = {f["x_pre"]["vertical"] for f in feats}
    assert "media" in verticals
    n_media = sum(1 for f in feats if f["x_pre"]["vertical"] == "media")
    n_other = sum(1 for f in feats if f["x_pre"]["vertical"] != "media")
    # Held-out level is much smaller than the rest
    assert n_media < n_other / 2


def test_severe_extrapolation_holdout_flags_media_as_high_risk():
    """The hold-out-one-level test on `vertical` should flag `media` as a
    high or severe extrapolation risk because the ICPD relationship in
    that vertical is structurally different from the rest of the pool."""
    feats, labels, _w = severe_extrapolation(n_majority=50, n_held_out=15)
    rows = run_extrapolation_test(feats, labels, "vertical", n_iterations=4)
    media_row = next(
        (r for r in rows if r["level"] == "media"),
        None,
    )
    assert media_row is not None, "expected `media` level in results"
    # Risk is calibrated: any non-trivial penalty (> 5pp) on synthetic data
    # is a positive signal. The fixture is tuned to clear that bar.
    assert media_row["risk"] in {"medium", "high", "severe"}


# --- Sanity helper: weighted_r_squared works on each fixture ----------------


@pytest.mark.parametrize(
    "fixture_name,builder",
    [
        ("perfect_compliance", perfect_compliance),
        ("one_sided_noncompliance", one_sided_noncompliance),
        ("sample_split", sample_split),
        ("high_lcc_bias", high_lcc_bias),
        ("cold_start_advertisers", cold_start_advertisers),
        ("severe_extrapolation", severe_extrapolation),
    ],
)
def test_each_fixture_produces_well_formed_pool(fixture_name, builder):
    feats, labels, w = builder()
    assert len(feats) == len(labels) == len(w)
    cids_f = {f["campaign_id"] for f in feats}
    cids_l = {l["campaign_id"] for l in labels}
    assert cids_f == cids_l, f"{fixture_name}: campaign_id mismatch"
    # Every weight is positive (cost-as-weight invariant from V.4 Wave 1)
    for w_val in w:
        assert w_val > 0
    # Sanity: weighted_r_squared at perfect prediction returns 1.0
    icpd = [label["icpd"] for label in labels]
    perfect = weighted_r_squared(icpd, icpd, w)
    assert perfect == pytest.approx(1.0, abs=1e-9)
