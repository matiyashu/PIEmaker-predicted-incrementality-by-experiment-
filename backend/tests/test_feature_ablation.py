"""Feature ablation tests — paper Figure 2 spec coverage (Prompt 2.3)."""

from __future__ import annotations

import random

import pytest

from ml.feature_ablation import ABLATION_SPECS, run_ablation


def _rows(n: int = 30) -> tuple[list[dict], list[dict], list[float]]:
    rng = random.Random(11)
    feats: list[dict] = []
    labels: list[dict] = []
    weights: list[float] = []
    for i in range(n):
        cid = f"CMP-{i:03d}"
        cpd = rng.uniform(0.5, 5.0)
        lcc7 = rng.uniform(0.05, 0.5)
        icpd_val = 0.3 * cpd + 1.2 * lcc7 + rng.gauss(0, 0.02)
        feats.append(
            {
                "campaign_id": cid,
                "x_pre": {
                    "objective": "conversions",
                    "vertical": rng.choice(["ecommerce", "travel"]),
                    "audience_type": "retargeting",
                    "funnel_stage": "lower",
                    "conversion_optimization": "yes",
                    "custom_audience": "yes",
                    "advertiser_platform_experience_months": 12,
                    "creative_format": "video",
                    "placement": "feed",
                    "bid_strategy": "lowest_cost",
                    "market": "US",
                    "spend_tier": "high",
                    "platform": "meta",
                    "month": 3,
                    "quarter": 1,
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
        weights.append(1.0)
    return feats, labels, weights


def test_ablation_returns_all_five_specs():
    feats, labels, weights = _rows(30)
    # V.4 ablation runs OOF + bootstrap CI; use cheap params for unit speed.
    rows = run_ablation(feats, labels, cost_weights=weights, n_splits=3, n_bootstrap=20)
    assert [r["spec"] for r in rows] == ABLATION_SPECS
    for r in rows:
        # V.4 shape: n_observations + n_splits + n_bootstrap + CI band.
        assert r["n_observations"] == 30
        assert r["n_bootstrap"] == 20
        assert r["n_splits"] >= 2
        assert "ci_lower" in r and "ci_upper" in r


def test_ablation_rejects_too_few_rows():
    feats, labels, _ = _rows(3)
    with pytest.raises(ValueError):
        run_ablation(feats, labels)


def test_ablation_rejects_mismatched_weights_length():
    feats, labels, _ = _rows(20)
    with pytest.raises(ValueError):
        run_ablation(feats, labels, cost_weights=[1.0, 1.0], n_splits=3, n_bootstrap=20)


def test_ablation_full_spec_at_least_as_strong_as_pre_only():
    """PIE(Full) should not have a worse weighted R² than PIE(Pre) on a clean
    synthetic donor pool where the post features carry the signal."""
    feats, labels, weights = _rows(60)
    rows = {
        r["spec"]: r["weighted_r2"]
        for r in run_ablation(feats, labels, weights, n_splits=4, n_bootstrap=20)
    }
    # Allow generous slack for RF stochasticity + OOF noise.
    assert rows["PIE(Full)"] >= rows["PIE(Pre)"] - 0.20
