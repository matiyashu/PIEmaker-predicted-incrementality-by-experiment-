"""Hold-out-one-level extrapolation tests (Prompt 2.4)."""

from __future__ import annotations

import random

import pytest

from ml.holdout_one_level import (
    RISK_BANDS,
    SEGMENTATION_VARS,
    _classify,
    run_extrapolation_test,
)


def _rows(n: int = 40) -> tuple[list[dict], list[dict]]:
    rng = random.Random(13)
    feats: list[dict] = []
    labels: list[dict] = []
    for i in range(n):
        cid = f"CMP-{i:03d}"
        vertical = rng.choice(["ecommerce", "travel", "finance"])
        cpd = rng.uniform(0.5, 5.0)
        lcc7 = rng.uniform(0.05, 0.5)
        icpd_val = 0.3 * cpd + 1.2 * lcc7 + rng.gauss(0, 0.02)
        feats.append(
            {
                "campaign_id": cid,
                "x_pre": {
                    "objective": "conversions",
                    "vertical": vertical,
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
    return feats, labels


def test_segmentation_vars_includes_paper_targets():
    for var in (
        "vertical",
        "conversion_optimization",
        "custom_audience",
        "advertiser_platform_experience_months",
    ):
        assert var in SEGMENTATION_VARS


def test_run_extrapolation_returns_one_row_per_level():
    feats, labels = _rows(40)
    rows = run_extrapolation_test(feats, labels, "vertical", n_iterations=3)
    levels = {r["level"] for r in rows}
    assert levels.issubset({"ecommerce", "travel", "finance"})
    for r in rows:
        assert r["segmentation_var"] == "vertical"
        assert r["risk"] in {b[0] for b in RISK_BANDS} | {"unknown"}


def test_run_extrapolation_rejects_unknown_var():
    feats, labels = _rows(40)
    with pytest.raises(ValueError):
        run_extrapolation_test(feats, labels, "platform_x", n_iterations=2)


def test_run_extrapolation_rejects_too_few_rows():
    feats, labels = _rows(4)
    with pytest.raises(ValueError):
        run_extrapolation_test(feats, labels, "vertical", n_iterations=2)


def test_classify_risk_thresholds():
    # Bands: severe ≥25, high ≥15, medium ≥5, low otherwise
    assert _classify(30.0) == "severe"
    assert _classify(20.0) == "high"
    assert _classify(10.0) == "medium"
    assert _classify(2.0) == "low"
    assert _classify(-1.0) == "low"
