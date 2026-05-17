"""V.4 Wave 2 — ablation OOF/CI, sample-size curve, LCC calibration by segment,
advertiser CV, bootstrap over advertisers, hold-out full distributions."""

from __future__ import annotations

import math
import random

import pytest

from ml.advertiser_cv import run_existing_vs_new
from ml.bootstrap_advertisers import bootstrap_over_advertisers
from ml.feature_ablation import ABLATION_SPECS, run_ablation
from ml.holdout_one_level import run_extrapolation_test
from ml.lcc_calibration_by_segment import calibration_by_segment
from ml.sample_size_curve import DEFAULT_SIZES, run_pool_size_curve


def _synth_pool(n: int = 60, n_advertisers: int = 12, seed: int = 5) -> tuple[
    list[dict], list[dict]
]:
    """Reusable synthetic donor pool with realistic structure for V.4 tests.

    Multiple campaigns per advertiser (so advertiser_cv has cohorts), 3 calendar
    years, mixed verticals/audiences, well-correlated ICPD signal.
    """
    rng = random.Random(seed)
    feats: list[dict] = []
    labels: list[dict] = []
    for i in range(n):
        cid = f"CMP-{i:03d}"
        adv_idx = i % n_advertisers  # multiple campaigns per advertiser
        cpd = rng.uniform(0.5, 5.0)
        lcc7 = rng.uniform(0.05, 0.5)
        icpd_val = 0.3 * cpd + 1.2 * lcc7 + rng.gauss(0, 0.02)
        feats.append(
            {
                "campaign_id": cid,
                "cost": rng.uniform(20_000, 200_000),
                "x_pre": {
                    "objective": "conversions",
                    "vertical": rng.choice(["ecommerce", "travel", "finance"]),
                    "audience_type": rng.choice(["retargeting", "prospecting"]),
                    "funnel_stage": "lower",
                    "conversion_optimization": rng.choice(["yes", "no"]),
                    "custom_audience": rng.choice(["yes", "no"]),
                    "advertiser_platform_experience_months": rng.randint(6, 36),
                    "advertiser_id": f"ADV-{adv_idx:03d}",
                    "advertiser_size": rng.choice(["smb", "mid_market", "enterprise"]),
                    "campaign_year": rng.choice([2024, 2025, 2026]),
                    "creative_format": "video",
                    "placement": "feed",
                    "bid_strategy": "lowest_cost",
                    "market": "US",
                    "spend_tier": "high",
                    "platform": "meta",
                    "month": rng.randint(1, 12),
                    "quarter": rng.randint(1, 4),
                    "campaign_duration_days": 21,
                },
                "x_post": {
                    "exposure_rate": 0.8,
                    "ctr": 0.02,
                    "clicks_per_dollar": 2.0,
                    "conversions_per_dollar": cpd,
                    "lcc_1h_per_dollar": 0.01,
                    "lcc_1d_per_dollar": 0.02,
                    "lcc_7d_per_dollar": lcc7,
                    "lcc_28d_per_dollar": 0.08,
                    "view_through_per_dollar": 0.01,
                    "avg_dwell_time": 10.0,
                },
            }
        )
        labels.append({"campaign_id": cid, "icpd": icpd_val})
    return feats, labels


# --- Phase 3a: ablation now returns OOF + 1000-bootstrap CI -------------------


def test_ablation_returns_one_row_per_spec_with_ci_band():
    feats, labels = _synth_pool(40, seed=11)
    rows = run_ablation(feats, labels, n_splits=4, n_bootstrap=50)
    assert [r["spec"] for r in rows] == list(ABLATION_SPECS)
    for r in rows:
        assert {"weighted_r2", "ci_lower", "ci_upper", "ci_mean"}.issubset(r)
        assert r["n_bootstrap"] == 50
        # CI must contain the point estimate (or be NaN-degenerate on tiny pools)
        if math.isfinite(r["ci_lower"]) and math.isfinite(r["weighted_r2"]):
            assert r["ci_lower"] <= r["weighted_r2"] + 1e-6
            assert r["weighted_r2"] - 1e-6 <= r["ci_upper"]


def test_ablation_full_spec_no_worse_than_pre_only():
    feats, labels = _synth_pool(60, seed=13)
    rows = {r["spec"]: r["weighted_r2"] for r in run_ablation(feats, labels, n_splits=4, n_bootstrap=50)}
    # On synthetic data where lcc/cpd carry the signal, PIE(Full) should
    # match or exceed PIE(Pre); allow generous slack for RF stochasticity.
    assert rows["PIE(Full)"] >= rows["PIE(Pre)"] - 0.20


# --- Phase 3b: sample-size curve ----------------------------------------------


def test_pool_size_curve_returns_points_at_requested_sizes():
    feats, labels = _synth_pool(60, seed=17)
    points = run_pool_size_curve(
        feats, labels, sizes=[20, 40], n_subsamples=2, n_splits=3
    )
    assert len(points) == 2
    for p in points:
        assert {"pool_size", "weighted_r2_median", "weighted_r2_p025", "weighted_r2_p975"}.issubset(p)
        assert p["weighted_r2_p025"] <= p["weighted_r2_median"] <= p["weighted_r2_p975"] + 1e-9


def test_pool_size_curve_skips_sizes_above_pool_size():
    feats, labels = _synth_pool(30, seed=18)
    points = run_pool_size_curve(
        feats, labels, sizes=[20, 1000], n_subsamples=2, n_splits=3
    )
    assert [p["pool_size"] for p in points] == [20]


def test_pool_size_curve_default_sizes_constant():
    assert DEFAULT_SIZES[0] < DEFAULT_SIZES[-1]
    assert sorted(DEFAULT_SIZES) == DEFAULT_SIZES


# --- Phase 3c: LCC calibration by segment -------------------------------------


def test_calibration_by_segment_returns_one_row_per_level():
    feats, labels = _synth_pool(60, seed=21)
    rows = calibration_by_segment(feats, labels, "vertical", min_n_per_level=3)
    levels = {r["level"] for r in rows}
    assert levels  # at least one level passes the min_n filter
    for r in rows:
        assert r["n"] >= 3
        # Residual percentiles are well-ordered
        assert r["residual_p10"] <= r["residual_mean"] <= r["residual_p90"] + 1e-9


def test_calibration_by_segment_rejects_unknown_var():
    feats, labels = _synth_pool(30, seed=22)
    with pytest.raises(ValueError):
        calibration_by_segment(feats, labels, "platform_x_unknown")


def test_calibration_by_segment_rejects_missing_lcc():
    feats, labels = _synth_pool(30, seed=23)
    # Strip lcc_7d_per_dollar from every row
    for f in feats:
        f["x_post"].pop("lcc_7d_per_dollar", None)
    with pytest.raises(ValueError):
        calibration_by_segment(feats, labels, "vertical")


# --- Phase 4a: hold-out distributions persisted -------------------------------


def test_holdout_returns_full_distributions():
    feats, labels = _synth_pool(40, seed=31)
    rows = run_extrapolation_test(feats, labels, "vertical", n_iterations=4)
    assert rows, "should return at least one level"
    for r in rows:
        assert "within_r2_dist" in r
        assert "extrapolation_r2_dist" in r
        assert "penalty_pp_dist" in r
        # Distribution lists are non-empty when paired iterations succeeded
        assert isinstance(r["within_r2_dist"], list)
        assert isinstance(r["extrapolation_r2_dist"], list)
        # p10/p90 tail fields exist
        assert "within_r2_p10" in r
        assert "extrapolation_r2_p90" in r


# --- Phase 4b: existing-vs-new advertiser CV ----------------------------------


def test_advertiser_cv_returns_two_cohorts():
    # Build a pool with mixed cohorts: some advertisers have 1 campaign (new),
    # others have multiple campaigns (existing).
    rng = random.Random(101)
    feats: list[dict] = []
    labels: list[dict] = []
    # 6 advertisers with 5 campaigns each (existing) + 10 advertisers with 1 each (new)
    cid_n = 0
    for adv in range(6):
        for _ in range(5):
            feats.append(_mk(cid_n, f"ADV-EXIST-{adv:02d}", rng))
            labels.append({"campaign_id": f"CMP-{cid_n:03d}", "icpd": rng.uniform(0.02, 0.15)})
            cid_n += 1
    for adv in range(10):
        feats.append(_mk(cid_n, f"ADV-NEW-{adv:02d}", rng))
        labels.append({"campaign_id": f"CMP-{cid_n:03d}", "icpd": rng.uniform(0.02, 0.15)})
        cid_n += 1

    result = run_existing_vs_new(feats, labels, n_splits=3)
    cohorts = {c["cohort"]: c for c in result["cohorts"]}
    assert {"existing", "new"} == set(cohorts.keys())
    assert cohorts["existing"]["n"] > 0
    assert cohorts["new"]["n"] > 0
    assert "cohort_gap_pp" in result


def test_advertiser_cv_rejects_single_advertiser():
    rng = random.Random(102)
    feats: list[dict] = []
    labels: list[dict] = []
    for i in range(12):
        feats.append(_mk(i, "ADV-ONLY", rng))
        labels.append({"campaign_id": f"CMP-{i:03d}", "icpd": rng.uniform(0.02, 0.15)})
    with pytest.raises(ValueError):
        run_existing_vs_new(feats, labels)


# --- Phase 4c: bootstrap over advertisers -------------------------------------


def test_bootstrap_advertisers_returns_distribution():
    feats, labels = _synth_pool(80, n_advertisers=10, seed=41)
    result = bootstrap_over_advertisers(feats, labels, n_draws=8, seed=1)
    assert result["n_advertisers"] == 10
    assert result["n_draws"] > 0
    assert result["p025"] <= result["mean"] <= result["p975"] + 1e-9
    assert len(result["distribution"]) == result["n_draws"]


def test_bootstrap_advertisers_rejects_too_few_advertisers():
    feats, labels = _synth_pool(30, n_advertisers=2, seed=42)
    with pytest.raises(ValueError):
        bootstrap_over_advertisers(feats, labels, n_draws=5)


# --- helpers ------------------------------------------------------------------


def _mk(cid_n: int, adv_id: str, rng: random.Random) -> dict:
    return {
        "campaign_id": f"CMP-{cid_n:03d}",
        "cost": rng.uniform(20_000, 200_000),
        "x_pre": {
            "objective": "conversions",
            "vertical": rng.choice(["ecommerce", "travel"]),
            "audience_type": "retargeting",
            "funnel_stage": "lower",
            "conversion_optimization": "yes",
            "custom_audience": "yes",
            "advertiser_platform_experience_months": 12,
            "advertiser_id": adv_id,
            "advertiser_size": rng.choice(["smb", "mid_market"]),
            "campaign_year": rng.choice([2024, 2025]),
            "creative_format": "video",
            "placement": "feed",
            "bid_strategy": "lowest_cost",
            "market": "US",
            "spend_tier": "medium",
            "platform": "meta",
            "month": 3,
            "quarter": 1,
            "campaign_duration_days": 21,
        },
        "x_post": {
            "exposure_rate": 0.8,
            "ctr": 0.02,
            "clicks_per_dollar": 2.0,
            "conversions_per_dollar": rng.uniform(0.5, 5.0),
            "lcc_1h_per_dollar": 0.01,
            "lcc_1d_per_dollar": 0.02,
            "lcc_7d_per_dollar": rng.uniform(0.05, 0.5),
            "lcc_28d_per_dollar": 0.08,
            "view_through_per_dollar": 0.01,
            "avg_dwell_time": 10.0,
        },
    }
