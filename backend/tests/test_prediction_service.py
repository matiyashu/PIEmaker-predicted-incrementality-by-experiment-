"""Prediction service tests (Prompt 3.1)."""

from __future__ import annotations

import random

import pytest

from ml import model_registry
from ml.train_random_forest import train
from services import prediction_service as ps
from services.persistence import reset, upsert


@pytest.fixture(autouse=True)
def _reset_state():
    reset()
    for p in model_registry._REGISTRY_DIR.glob("*.pkl"):
        p.unlink()
    yield
    reset()
    for p in model_registry._REGISTRY_DIR.glob("*.pkl"):
        p.unlink()


def _train_and_register(status: str = "production") -> dict:
    """Synthesize features+labels, train an RF, register a model."""
    rng = random.Random(101)
    feats: list[dict] = []
    labels: list[dict] = []
    for i in range(30):
        cid = f"CMP-{i:03d}"
        cpd = rng.uniform(0.5, 5.0)
        lcc7 = rng.uniform(0.05, 0.5)
        icpd_val = 0.3 * cpd + 1.2 * lcc7 + rng.gauss(0, 0.02)
        feats.append({
            "campaign_id": cid,
            "x_pre": {
                "objective": "conversions",
                "vertical": rng.choice(["ecommerce", "travel", "finance"]),
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
        })
        labels.append({"campaign_id": cid, "icpd": icpd_val})

    artifacts = train(feats, labels, n_splits=3)
    rec = model_registry.register_model(
        name="pred-test-rf",
        algorithm="random_forest",
        feature_set_version="v1",
        hyperparameters=artifacts.chosen_hyperparameters,
        training_donor_pool_size=400 if status == "production" else 250,
        estimator=artifacts.estimator,
        status=status,
    )
    # Persist a bootstrap CI metric so the prediction service has a band
    model_registry.record_metric(
        rec["id"],
        "weighted_r_squared_bootstrap_mean",
        0.80,
        ci_lower=0.74,
        ci_upper=0.86,
    )
    return rec


def _campaign_spec() -> dict:
    return {
        "campaign_id": "PRED-CMP-NEW",
        "objective": "conversions",
        "vertical": "ecommerce",
        "audience_type": "retargeting",
        "funnel_stage": "lower",
        "conversion_optimization": "yes",
        "custom_audience": "yes",
        "advertiser_platform_experience_months": 18,
        "creative_format": "video",
        "placement": "feed",
        "bid_strategy": "lowest_cost",
        "market": "US",
        "spend_tier": "high",
        "platform": "meta",
        "start_date": "2026-06-01",
        "end_date": "2026-06-22",
        "cost": 75_000,
        "test_users": 1_500_000,
        "exposed_test_users": 1_200_000,
        "clicks": 30_000,
        "impressions": 4_000_000,
        "conversions": 28_000,
        "lcc_7d": 5_000,
    }


def test_score_returns_predicted_icpd_with_ci():
    rec = _train_and_register("production")
    out = ps.score_campaign(_campaign_spec(), model_id=rec["id"])
    assert "predicted_icpd" in out
    assert isinstance(out["predicted_icpd"], float)
    assert out["model_version_id"] == rec["id"]
    assert out["ci_lower"] is not None and out["ci_upper"] is not None
    assert out["ci_upper"] > out["ci_lower"]
    assert out["watermark"] is None  # production model


def test_score_research_model_is_watermarked():
    rec = _train_and_register("research")
    out = ps.score_campaign(_campaign_spec(), model_id=rec["id"])
    assert out["model_status"] == "research"
    assert out["watermark"] is not None
    assert "advisory" in out["watermark"].lower()


def test_score_no_models_raises():
    with pytest.raises(ps.PredictionError):
        ps.score_campaign(_campaign_spec())


def test_score_unknown_model_id_raises():
    _train_and_register("production")
    with pytest.raises(ps.PredictionError):
        ps.score_campaign(_campaign_spec(), model_id="bogus-id")


def test_default_selects_production_over_research():
    research = _train_and_register("research")
    prod = _train_and_register("production")
    out = ps.score_campaign(_campaign_spec())
    assert out["model_version_id"] == prod["id"]
    assert out["model_version_id"] != research["id"]


def test_segment_risk_attached_when_holdout_persisted():
    rec = _train_and_register("production")
    upsert(
        "holdout_results",
        {
            "id": "vertical|ecommerce",
            "segmentation_var": "vertical",
            "level": "ecommerce",
            "within_r2_median": 0.80,
            "extrapolation_r2_median": 0.55,
            "penalty_pp": 25.0,
            "n_iterations": 20,
            "risk": "severe",
            "created_at": "2026-05-07T00:00:00Z",
        },
        key="id",
    )
    out = ps.score_campaign(_campaign_spec(), model_id=rec["id"])
    risks = out["segment_risks"]
    assert any(
        r["segmentation_var"] == "vertical" and r["risk"] == "severe" for r in risks
    )
    assert out["worst_segment_risk"]["risk"] == "severe"


def test_run_persisted_and_listable():
    rec = _train_and_register("production")
    out = ps.score_campaign(_campaign_spec(), model_id=rec["id"])
    runs = ps.list_runs()
    assert any(r["id"] == out["id"] for r in runs)
    fetched = ps.get_run(out["id"])
    assert fetched is not None
    assert fetched["predicted_icpd"] == out["predicted_icpd"]
