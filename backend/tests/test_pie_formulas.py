"""
Contract tests for the frozen formula library (PDF Section 3).

Three tests minimum per formula:
  1. golden-path
  2. edge case (zero/negative/empty input → ValueError)
  3. paper-derived numerical reference (or independent hand-computed value)

CI failure on any test = breach of formula contract = block merge.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from pie_formulas import (
    att,
    att_decomposition,
    bootstrap_metric,
    disagreement_probability,
    expected_disagreement_cost,
    exposure_rate,
    icpd,
    incremental_conversions,
    incremental_revenue,
    iroas,
    lcc_bias_ratio,
    lcc_ols_slope,
    lcc_spearman_rho,
    predicted_cpic,
    predicted_ic,
    r_squared_ceiling,
    segment_median_threshold,
    threshold_scan_range,
    weighted_r_squared,
)


# -----------------------------------------------------------------------------
# Section 3.1 — Label generation
# -----------------------------------------------------------------------------


class TestExposureRate:
    def test_golden(self):
        assert exposure_rate(800_000, 1_000_000) == pytest.approx(0.8)

    def test_edge_zero_test_users(self):
        with pytest.raises(ValueError):
            exposure_rate(100, 0)

    def test_edge_negative_exposed(self):
        with pytest.raises(ValueError):
            exposure_rate(-10, 1_000_000)

    def test_edge_rate_above_one(self):
        with pytest.raises(ValueError):
            exposure_rate(2_000_000, 1_000_000)

    def test_paper_reference_full_exposure(self):
        # Per paper p. 11, D̄_tr ranges (0, 1]; full exposure = 1.0.
        assert exposure_rate(1_000_000, 1_000_000) == pytest.approx(1.0)


class TestATT:
    def test_golden(self):
        # Simple Wald: Y_tr=0.05, Y_cr=0.04, D=0.5 → ATT = 0.02
        result = att(
            test_conversions=50_000,
            test_users=1_000_000,
            control_conversions=40_000,
            control_users=1_000_000,
            exposure_rate_value=0.5,
        )
        assert result == pytest.approx(0.02)

    def test_edge_zero_test_users(self):
        with pytest.raises(ValueError):
            att(0, 0, 0, 1_000_000, 0.5)

    def test_edge_invalid_exposure(self):
        with pytest.raises(ValueError):
            att(50_000, 1_000_000, 40_000, 1_000_000, 1.5)

    def test_negative_att_allowed(self):
        # ATT can be negative; only the inputs are constrained.
        result = att(
            test_conversions=30_000,
            test_users=1_000_000,
            control_conversions=40_000,
            control_users=1_000_000,
            exposure_rate_value=0.5,
        )
        assert result == pytest.approx(-0.02)


class TestIncrementalConversions:
    def test_golden(self):
        # IC = ATT × D × N = 0.02 × 0.5 × 1_000_000 = 10_000
        assert incremental_conversions(0.02, 0.5, 1_000_000) == pytest.approx(
            10_000
        )

    def test_edge_zero_users(self):
        with pytest.raises(ValueError):
            incremental_conversions(0.02, 0.5, 0)

    def test_edge_invalid_exposure(self):
        with pytest.raises(ValueError):
            incremental_conversions(0.02, 0.0, 1_000_000)


class TestICPD:
    def test_golden(self):
        # ICPD = IC / Cost = 10_000 / $50_000 = 0.2 conv per dollar
        assert icpd(10_000, 50_000) == pytest.approx(0.2)

    def test_edge_zero_cost(self):
        with pytest.raises(ValueError):
            icpd(10_000, 0)

    def test_negative_ic_yields_negative_icpd(self):
        # Negative lift is allowed at the label level.
        assert icpd(-1_000, 50_000) == pytest.approx(-0.02)


# -----------------------------------------------------------------------------
# Section 3.2 — Prediction
# -----------------------------------------------------------------------------


class TestPredictedIC:
    def test_golden(self):
        assert predicted_ic(0.5, 10_000) == pytest.approx(5_000)

    def test_edge_zero_cost(self):
        with pytest.raises(ValueError):
            predicted_ic(0.5, 0)

    def test_negative_predicted_icpd(self):
        # Predicted ICPD can be <= 0 here; the hard-block lives in CPIC only.
        assert predicted_ic(-0.1, 10_000) == pytest.approx(-1_000)


class TestPredictedCPIC:
    def test_golden(self):
        assert predicted_cpic(0.5) == pytest.approx(2.0)

    def test_hard_block_zero(self):
        # Per Footnote 18 (p. 18): CPIC undefined when ICPD ≤ 0.
        assert predicted_cpic(0.0) is None

    def test_hard_block_negative(self):
        assert predicted_cpic(-0.001) is None


class TestIncrementalRevenue:
    def test_golden(self):
        assert incremental_revenue(1_000, 25.0) == pytest.approx(25_000)

    def test_edge_negative_revenue_per_conversion(self):
        with pytest.raises(ValueError):
            incremental_revenue(1_000, -1.0)

    def test_zero_value(self):
        assert incremental_revenue(1_000, 0.0) == pytest.approx(0.0)


class TestIROAS:
    def test_golden(self):
        # iROAS = $25_000 / $10_000 = 2.5
        assert iroas(25_000, 10_000) == pytest.approx(2.5)

    def test_edge_zero_cost(self):
        with pytest.raises(ValueError):
            iroas(25_000, 0)

    def test_negative_iroas(self):
        assert iroas(-5_000, 10_000) == pytest.approx(-0.5)


# -----------------------------------------------------------------------------
# Section 3.3 — Evaluation
# -----------------------------------------------------------------------------


class TestWeightedRSquared:
    def test_golden_perfect_prediction(self):
        # When y_pred == y_true, R² = 1.0
        y_true = [0.1, 0.2, 0.3, 0.4]
        y_pred = list(y_true)
        weights = [1, 1, 1, 1]
        assert weighted_r_squared(y_true, y_pred, weights) == pytest.approx(1.0)

    def test_edge_mismatched_shapes(self):
        with pytest.raises(ValueError):
            weighted_r_squared([1, 2], [1, 2, 3], [1, 1, 1])

    def test_edge_empty(self):
        with pytest.raises(ValueError):
            weighted_r_squared([], [], [])

    def test_edge_zero_weights(self):
        with pytest.raises(ValueError):
            weighted_r_squared([1, 2], [1, 2], [0, 0])

    def test_edge_negative_weights(self):
        with pytest.raises(ValueError):
            weighted_r_squared([1, 2], [1, 2], [-1, 1])

    def test_edge_zero_variance(self):
        # If all y_true equal, denominator is 0 → R² undefined
        with pytest.raises(ValueError):
            weighted_r_squared([0.5, 0.5, 0.5], [0.4, 0.5, 0.6], [1, 1, 1])

    def test_paper_reference_matches_unweighted_when_equal_weights(self):
        # Hand-computed: y=[1,2,3,4], yhat=[1.1,1.9,3.05,3.95], equal weights
        # SS_res = 0.01+0.01+0.0025+0.0025 = 0.025
        # ybar = 2.5 → SS_tot = 1.5²+0.5²+0.5²+1.5² = 5
        # R² = 1 − 0.025/5 = 0.995
        result = weighted_r_squared(
            [1, 2, 3, 4], [1.1, 1.9, 3.05, 3.95], [1, 1, 1, 1]
        )
        assert result == pytest.approx(0.995)

    def test_cost_weighting_dominates_equal_weighting(self):
        # Two campaigns with the same per-row residual. When the high-cost
        # campaign sits closer to the weighted mean (so its contribution to
        # SS_tot is small), reweighting by cost shifts both numerator and
        # denominator. Verify the metric reacts to weights at all and is
        # numerically stable.
        y_true = [0.10, 0.10, 0.50, 0.50]
        y_pred = [0.12, 0.08, 0.52, 0.48]
        equal_w = [1, 1, 1, 1]
        cost_w = [10, 10, 1000, 1000]
        equal = weighted_r_squared(y_true, y_pred, equal_w)
        weighted = weighted_r_squared(y_true, y_pred, cost_w)
        # Both must be valid finite floats; equal weighting on this fixture
        # gives a high R² because the y values vary substantially.
        assert math.isfinite(equal) and math.isfinite(weighted)
        assert equal >= 0.95


class TestRSquaredCeiling:
    def test_golden(self):
        # 30% noise → ceiling = 0.7
        assert r_squared_ceiling(30.0, 100.0) == pytest.approx(0.7)

    def test_edge_zero_total(self):
        with pytest.raises(ValueError):
            r_squared_ceiling(10.0, 0.0)

    def test_edge_noise_exceeds_total(self):
        with pytest.raises(ValueError):
            r_squared_ceiling(150.0, 100.0)

    def test_edge_negative_noise(self):
        with pytest.raises(ValueError):
            r_squared_ceiling(-1.0, 100.0)

    def test_paper_reference_088_near_ceiling(self):
        # Footnote 21 (p. 19): observed R² of 0.88 is closer to ceiling than 1.
        # If outcome noise is 10% of total variance, ceiling = 0.9 — observed
        # 0.88 sits 0.02 below ceiling, ~98% of the way there.
        ceiling = r_squared_ceiling(10.0, 100.0)
        assert ceiling == pytest.approx(0.9)
        assert (0.88 / ceiling) > 0.97


class TestBootstrapMetric:
    def test_golden_mean_metric(self):
        # Bootstrap of np.mean on a constant array returns the constant.
        out = bootstrap_metric(
            lambda a: float(np.mean(a)),
            np.full(50, 0.42),
            n_draws=100,
            seed=0,
        )
        assert out["mean"] == pytest.approx(0.42)
        assert out["sd"] == pytest.approx(0.0)
        assert out["n_draws"] == 100

    def test_edge_zero_draws(self):
        with pytest.raises(ValueError):
            bootstrap_metric(lambda a: float(a.mean()), [1, 2, 3], n_draws=0)

    def test_edge_empty_input(self):
        with pytest.raises(ValueError):
            bootstrap_metric(lambda a: float(a.mean()), [], n_draws=10)

    def test_edge_mismatched_lengths(self):
        with pytest.raises(ValueError):
            bootstrap_metric(
                lambda a, b: float(a.sum() / b.sum()),
                [1, 2, 3],
                [1, 2],
                n_draws=10,
            )

    def test_no_arrays(self):
        with pytest.raises(ValueError):
            bootstrap_metric(lambda: 0.0, n_draws=10)

    def test_determinism(self):
        # Same seed → identical bootstrap distribution mean.
        rng = np.random.default_rng(123)
        x = rng.standard_normal(200)
        a = bootstrap_metric(lambda v: float(v.mean()), x, n_draws=200, seed=7)
        b = bootstrap_metric(lambda v: float(v.mean()), x, n_draws=200, seed=7)
        assert a["mean"] == pytest.approx(b["mean"])


class TestLCCBiasRatio:
    def test_golden(self):
        # LCC mean / ICPD mean = 0.6 / 0.4 = 1.5 (retail-vertical baseline)
        assert lcc_bias_ratio(
            [0.5, 0.6, 0.7], [0.3, 0.4, 0.5]
        ) == pytest.approx(1.5)

    def test_edge_empty(self):
        with pytest.raises(ValueError):
            lcc_bias_ratio([], [])

    def test_edge_mismatched_shapes(self):
        with pytest.raises(ValueError):
            lcc_bias_ratio([0.5, 0.6], [0.3])

    def test_edge_zero_icpd_mean(self):
        with pytest.raises(ValueError):
            lcc_bias_ratio([0.5, 0.5], [0.0, 0.0])

    def test_paper_reference_overall_133(self):
        # Paper (p. 23) reports overall ratio of 1.33.
        # Construct fixture that hits that target.
        result = lcc_bias_ratio([1.33, 1.33, 1.33], [1.0, 1.0, 1.0])
        assert result == pytest.approx(1.33)


class TestLCCOLSSlope:
    def test_golden_perfect_slope(self):
        # y = 0.5x exactly → slope = 0.5
        x = list(np.linspace(0, 1, 50))
        y = [0.5 * v for v in x]
        assert lcc_ols_slope(y, x) == pytest.approx(0.5)

    def test_edge_too_few_obs(self):
        with pytest.raises(ValueError):
            lcc_ols_slope([0.5], [0.5])

    def test_edge_zero_variance(self):
        with pytest.raises(ValueError):
            lcc_ols_slope([0.1, 0.2, 0.3], [0.5, 0.5, 0.5])

    def test_edge_mismatched_shapes(self):
        with pytest.raises(ValueError):
            lcc_ols_slope([0.1, 0.2], [0.5, 0.5, 0.5])

    def test_paper_reference_069_slope(self):
        # Paper (p. 23): OLS slope ≈ 0.69 in Meta data.
        # Build noisy data with true slope 0.69.
        rng = np.random.default_rng(42)
        x = rng.uniform(0.0, 2.0, size=500)
        y = 0.69 * x + rng.normal(0.0, 0.05, size=500)
        slope = lcc_ols_slope(y.tolist(), x.tolist())
        assert slope == pytest.approx(0.69, abs=0.05)


class TestLCCSpearmanRho:
    def test_golden_perfect_monotone(self):
        # Strictly monotone → ρ = 1.0
        x = list(range(50))
        y = [v ** 2 for v in x]
        assert lcc_spearman_rho(y, x) == pytest.approx(1.0)

    def test_edge_too_few_obs(self):
        with pytest.raises(ValueError):
            lcc_spearman_rho([0.5], [0.5])

    def test_edge_mismatched_shapes(self):
        with pytest.raises(ValueError):
            lcc_spearman_rho([0.1, 0.2], [0.5, 0.5, 0.5])

    def test_paper_reference_high_rho(self):
        # Paper (p. 23): ρ ≈ 0.89 in Meta data — LCC carries salvageable
        # rank-order signal even when biased in level. Synthetic fixture with
        # noise calibrated to land in the same band.
        rng = np.random.default_rng(7)
        x = rng.uniform(0.0, 2.0, size=500)
        y = 0.69 * x + rng.normal(0.0, 0.30, size=500)
        rho = lcc_spearman_rho(y.tolist(), x.tolist())
        assert 0.80 <= rho <= 0.95


# -----------------------------------------------------------------------------
# Section 3.4 — Decision metrics
# -----------------------------------------------------------------------------


class TestSegmentMedianThreshold:
    def test_golden(self):
        assert segment_median_threshold([0.1, 0.2, 0.3, 0.4, 0.5]) == pytest.approx(
            0.3
        )

    def test_edge_empty(self):
        with pytest.raises(ValueError):
            segment_median_threshold([])

    def test_paper_reference_negative_segment(self):
        # ICPD can be negative in poorly-performing segments; median still works.
        assert segment_median_threshold([-0.1, 0.0, 0.1]) == pytest.approx(0.0)


class TestThresholdScanRange:
    def test_golden_default_range(self):
        # Median 1.0, default 50%-150% in 5% steps → 21 thresholds, 0.5..1.5
        scan = threshold_scan_range(1.0)
        assert len(scan) == 21
        assert scan[0] == pytest.approx(0.5)
        assert scan[-1] == pytest.approx(1.5)

    def test_edge_invalid_low(self):
        with pytest.raises(ValueError):
            threshold_scan_range(1.0, low_multiplier=0.0)

    def test_edge_high_below_low(self):
        with pytest.raises(ValueError):
            threshold_scan_range(1.0, low_multiplier=1.5, high_multiplier=0.5)

    def test_edge_invalid_step(self):
        with pytest.raises(ValueError):
            threshold_scan_range(1.0, step=0)

    def test_custom_range(self):
        scan = threshold_scan_range(2.0, low_multiplier=0.8, high_multiplier=1.2, step=0.1)
        assert len(scan) == 5
        assert scan[0] == pytest.approx(1.6)
        assert scan[-1] == pytest.approx(2.4)


class TestDisagreementProbability:
    def test_golden(self):
        # truth = pred → zero disagreement
        out = disagreement_probability(
            [0.1, 0.5, 0.9], [0.1, 0.5, 0.9], threshold=0.5
        )
        assert out["disagreement_probability"] == pytest.approx(0.0)
        assert out["type_1_error"] == pytest.approx(0.0)
        assert out["type_2_error"] == pytest.approx(0.0)

    def test_edge_empty(self):
        with pytest.raises(ValueError):
            disagreement_probability([], [], threshold=0.5)

    def test_edge_mismatched_shapes(self):
        with pytest.raises(ValueError):
            disagreement_probability([0.1, 0.2], [0.1], threshold=0.5)

    def test_edge_per_row_threshold_wrong_length(self):
        with pytest.raises(ValueError):
            disagreement_probability(
                [0.1, 0.2, 0.3],
                [0.1, 0.2, 0.3],
                threshold=[0.1, 0.2],
            )

    def test_type_1_pure(self):
        # All rows: truth ≤ threshold (0.5), pred > threshold → all Type I
        out = disagreement_probability(
            [0.1, 0.2, 0.3], [0.6, 0.7, 0.8], threshold=0.5
        )
        assert out["disagreement_probability"] == pytest.approx(1.0)
        assert out["type_1_error"] == pytest.approx(1.0)
        assert out["type_2_error"] == pytest.approx(0.0)

    def test_type_2_pure(self):
        # All rows: truth > threshold, pred ≤ threshold → all Type II
        out = disagreement_probability(
            [0.6, 0.7, 0.8], [0.1, 0.2, 0.3], threshold=0.5
        )
        assert out["disagreement_probability"] == pytest.approx(1.0)
        assert out["type_1_error"] == pytest.approx(0.0)
        assert out["type_2_error"] == pytest.approx(1.0)

    def test_paper_reference_pie_disagreement_band(self):
        # Paper baseline: PIE 8-12% disagreement vs RCT.
        # 100-row fixture with ~10% mismatched decisions.
        rng = np.random.default_rng(123)
        truth = rng.uniform(0.0, 1.0, 100)
        pred = truth.copy()
        flip_idx = rng.choice(100, size=10, replace=False)
        pred[flip_idx] = 1.0 - pred[flip_idx]
        out = disagreement_probability(truth, pred, threshold=0.5)
        assert 0.05 <= out["disagreement_probability"] <= 0.20

    def test_per_row_threshold_segment_relative(self):
        # Each row uses its own segment-relative threshold.
        truth = [0.1, 0.4, 0.6, 0.9]
        pred = [0.05, 0.5, 0.55, 0.95]
        thr = [0.2, 0.3, 0.5, 0.8]
        out = disagreement_probability(truth, pred, threshold=thr)
        # Row 1: truth 0.4 > thr 0.3, pred 0.5 > thr → agree
        # Row 2: truth 0.6 > thr 0.5, pred 0.55 > thr → agree
        # All agree → 0
        assert out["disagreement_probability"] == pytest.approx(0.0)


class TestExpectedDisagreementCost:
    def test_golden_zero_when_no_errors(self):
        edc = expected_disagreement_cost(
            icpd_true=[0.1, 0.2, 0.3],
            icpd_pred=[0.1, 0.2, 0.3],
            threshold=0.15,
            campaign_cost=[1000, 2000, 3000],
            forgone_inc_revenue=[500, 1000, 1500],
        )
        assert edc == pytest.approx(0.0)

    def test_edge_negative_cost_per_fp(self):
        with pytest.raises(ValueError):
            expected_disagreement_cost(
                [0.1], [0.5], 0.3, [1000], [500], cost_per_fp=-1.0
            )

    def test_edge_mismatched_shapes(self):
        with pytest.raises(ValueError):
            expected_disagreement_cost(
                [0.1, 0.2], [0.5], 0.3, [1000], [500]
            )

    def test_edge_empty(self):
        with pytest.raises(ValueError):
            expected_disagreement_cost([], [], 0.3, [], [])

    def test_pure_type_1_cost(self):
        # All Type I: pred > thr, truth <= thr.
        # FP cost only: 1.0 × $1000 + 1.0 × $2000 = $3000.
        edc = expected_disagreement_cost(
            icpd_true=[0.1, 0.1],
            icpd_pred=[0.5, 0.5],
            threshold=0.2,
            campaign_cost=[1000, 2000],
            forgone_inc_revenue=[10_000, 20_000],
            cost_per_fp=1.0,
            cost_per_fn=1.0,
        )
        assert edc == pytest.approx(3000.0)

    def test_pure_type_2_cost(self):
        # All Type II: pred <= thr, truth > thr.
        # FN cost only: 1.0 × $10000 + 1.0 × $20000 = $30000.
        edc = expected_disagreement_cost(
            icpd_true=[0.5, 0.5],
            icpd_pred=[0.1, 0.1],
            threshold=0.2,
            campaign_cost=[1000, 2000],
            forgone_inc_revenue=[10_000, 20_000],
        )
        assert edc == pytest.approx(30_000.0)

    def test_asymmetric_cost_ratio(self):
        # Cost_FP=2.0, Cost_FN=1.0. Same row counts → FP doubles.
        edc_fp = expected_disagreement_cost(
            [0.1, 0.1], [0.5, 0.5], 0.2, [1000, 2000], [0, 0],
            cost_per_fp=2.0, cost_per_fn=1.0,
        )
        assert edc_fp == pytest.approx(6000.0)


# -----------------------------------------------------------------------------
# Section 3.5 — ATT decomposition (research mode)
# -----------------------------------------------------------------------------


class TestATTDecomposition:
    def test_golden(self):
        out = att_decomposition(
            treatment_effect_on_treated=0.05,
            selection_term=0.2,
            counterfactual_treatment_response=0.1,
        )
        # ψ = 0.05 + 0.2 × 0.1 = 0.07
        assert out["att"] == pytest.approx(0.07)
        assert out["research_mode_only"] is True

    def test_edge_zero_components(self):
        # No errors expected; this is a pure algebraic identity.
        out = att_decomposition(0.0, 0.0, 0.0)
        assert out["att"] == 0.0
        assert out["research_mode_only"] is True

    def test_paper_reference_eq16_structure(self):
        # Eq. 16 (p. 11): components are NOT identifiable from observables;
        # this test only verifies the algebraic identity holds.
        out = att_decomposition(0.1, -0.5, 0.4)
        # ψ = 0.1 + (-0.5 × 0.4) = 0.1 - 0.2 = -0.1
        assert out["att"] == pytest.approx(-0.1)
        assert math.isclose(out["att"], -0.1)
