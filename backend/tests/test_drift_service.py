"""Drift monitoring tests (Prompt 4.1)."""

from __future__ import annotations

import random

import pytest

from services import drift_service as ds
from services.feature_engineering_service import FEATURE_TABLE
from services.persistence import reset, upsert


@pytest.fixture(autouse=True)
def _reset_state():
    reset()
    yield
    reset()


def _persist_training(rows: list[dict]) -> None:
    for r in rows:
        upsert(FEATURE_TABLE, r, key="campaign_id")


def _make_row(
    cid: str,
    *,
    cpd: float,
    lcc7: float,
    vertical: str = "ecommerce",
    audience: str = "retargeting",
    mode: str = "training",
) -> dict:
    return {
        "campaign_id": cid,
        "feature_set_version": "v1",
        "mode": mode,
        "sample_id": None,
        "x_pre": {
            "objective": "conversions",
            "vertical": vertical,
            "audience_type": audience,
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
        "created_at": "2026-05-07T00:00:00Z",
    }


def _build_pool(n: int, *, cpd_lo=0.5, cpd_hi=5.0, lcc_lo=0.05, lcc_hi=0.5,
                verticals=("ecommerce", "travel", "finance"),
                seed: int = 11, mode="training") -> list[dict]:
    rng = random.Random(seed)
    return [
        _make_row(
            f"CMP-{seed}-{i:03d}",
            cpd=rng.uniform(cpd_lo, cpd_hi),
            lcc7=rng.uniform(lcc_lo, lcc_hi),
            vertical=rng.choice(verticals),
            mode=mode,
        )
        for i in range(n)
    ]


def test_no_drift_when_distributions_match():
    """With large enough samples drawn from identical distributions, the
    verdict should be stable. PSI is noisy on tiny samples, so we use 400/200."""
    train = _build_pool(400, seed=1, mode="training")
    _persist_training(train)
    score = _build_pool(200, seed=2, mode="scoring")
    result = ds.compute_drift(score)
    assert result["severity_counts"]["severe"] == 0
    assert result["verdict"] in ("stable", "watch")  # tolerate borderline noise


def test_severe_drift_triggers_retrain_recommendation():
    """Shift the scoring pool's CPD distribution far away from training."""
    train = _build_pool(50, seed=3, cpd_lo=0.5, cpd_hi=2.0,
                        lcc_lo=0.05, lcc_hi=0.10, mode="training")
    _persist_training(train)
    # Scoring pool uses a totally different range
    score = _build_pool(30, seed=4, cpd_lo=20.0, cpd_hi=40.0,
                        lcc_lo=2.0, lcc_hi=5.0, mode="scoring")
    result = ds.compute_drift(score)
    assert result["verdict"] == "retrain_recommended"
    assert result["severity_counts"]["severe"] >= 1
    # The two shifted features should be among the worst drifters
    top = {d["feature"] for d in result["drifts"][:3]}
    assert top & {"conversions_per_dollar", "lcc_7d_per_dollar"}


def test_categorical_drift_detected():
    train = _build_pool(60, seed=5, verticals=("ecommerce",), mode="training")
    _persist_training(train)
    score = _build_pool(20, seed=6, verticals=("travel",), mode="scoring")
    result = ds.compute_drift(score)
    vertical_drift = next(
        d for d in result["drifts"] if d["feature"] == "vertical"
    )
    assert vertical_drift["severity"] == "severe"
    assert vertical_drift["kind"] == "categorical"


def test_classify_thresholds():
    assert ds._classify(0.05) == "stable"
    assert ds._classify(0.15) == "moderate"
    assert ds._classify(0.30) == "severe"
    assert ds._classify(0.10) == "moderate"
    assert ds._classify(0.25) == "severe"


def test_no_training_rows_raises():
    score = _build_pool(10, mode="scoring")
    with pytest.raises(ValueError):
        ds.compute_drift(score)


def test_empty_scoring_raises():
    train = _build_pool(20, mode="training")
    _persist_training(train)
    with pytest.raises(ValueError):
        ds.compute_drift([])


def test_watch_verdict_with_three_moderate_drifters(monkeypatch):
    """If exactly 3 features land in moderate (no severe), verdict is 'watch'."""
    train = _build_pool(40, mode="training")
    _persist_training(train)
    score = _build_pool(20, mode="scoring")

    # Force the classifier so exactly 3 features land in 'moderate' and the
    # rest in 'stable'. This isolates the verdict-aggregation logic.
    counter = {"n": 0}

    def patched(psi: float) -> str:
        counter["n"] += 1
        if counter["n"] <= 3:
            return "moderate"
        return "stable"

    monkeypatch.setattr(ds, "_classify", patched)
    result = ds.compute_drift(score)
    assert result["verdict"] == "watch"
    assert result["severity_counts"]["moderate"] == 3
    assert result["severity_counts"]["severe"] == 0


def test_drift_results_sorted_descending_by_psi():
    train = _build_pool(50, seed=7, mode="training")
    _persist_training(train)
    score = _build_pool(20, seed=8, cpd_lo=10.0, cpd_hi=20.0, mode="scoring")
    result = ds.compute_drift(score)
    psis = [d["psi"] for d in result["drifts"]]
    assert psis == sorted(psis, reverse=True)
