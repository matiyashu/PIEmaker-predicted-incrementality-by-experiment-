"""Portfolio scoring tests (Prompt 3.2)."""

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
    rng = random.Random(202)
    feats: list[dict] = []
    labels: list[dict] = []
    for i in range(30):
        cid = f"CMP-{i:03d}"
        cpd = rng.uniform(0.5, 5.0)
        lcc7 = rng.uniform(0.05, 0.5)
        icpd = 0.3 * cpd + 1.2 * lcc7 + rng.gauss(0, 0.02)
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
        labels.append({"campaign_id": cid, "icpd": icpd})

    artifacts = train(feats, labels, n_splits=3)
    rec = model_registry.register_model(
        name="portfolio-test-rf",
        algorithm="random_forest",
        feature_set_version="v1",
        hyperparameters=artifacts.chosen_hyperparameters,
        training_donor_pool_size=400 if status == "production" else 250,
        estimator=artifacts.estimator,
        status=status,
    )
    model_registry.record_metric(
        rec["id"],
        "weighted_r_squared_bootstrap_mean",
        0.78,
        ci_lower=0.72,
        ci_upper=0.84,
    )
    return rec


def _portfolio(n: int = 5) -> list[dict]:
    rng = random.Random(303)
    out = []
    for i in range(n):
        out.append({
            "campaign_id": f"PORT-{i:03d}",
            "objective": "conversions",
            "vertical": rng.choice(["ecommerce", "travel", "finance"]),
            "audience_type": "retargeting",
            "funnel_stage": "lower",
            "conversion_optimization": "yes",
            "custom_audience": "yes",
            "advertiser_platform_experience_months": rng.randint(6, 36),
            "creative_format": "video",
            "placement": "feed",
            "bid_strategy": "lowest_cost",
            "market": "US",
            "spend_tier": "high",
            "platform": "meta",
            "start_date": "2026-06-01",
            "end_date": "2026-06-22",
            "cost": rng.uniform(20_000, 200_000),
            "test_users": rng.randint(800_000, 2_500_000),
            "exposed_test_users": rng.randint(600_000, 2_200_000),
            "clicks": rng.randint(20_000, 80_000),
            "impressions": rng.randint(2_000_000, 8_000_000),
            "conversions": rng.randint(15_000, 50_000),
            "lcc_7d": rng.randint(2_000, 8_000),
        })
    return out


def test_score_portfolio_returns_per_row_and_aggregates():
    rec = _train_and_register("production")
    out = ps.score_portfolio(_portfolio(5), model_id=rec["id"])
    assert len(out["runs"]) == 5
    for r in out["runs"]:
        assert isinstance(r["predicted_icpd"], float)
        assert r["model_version_id"] == rec["id"]
        assert r["ci_lower"] is not None and r["ci_upper"] is not None
        assert r["portfolio_run"] is True

    agg = out["aggregates"]
    assert agg["n"] == 5
    assert agg["p10_icpd"] <= agg["median_icpd"] <= agg["p90_icpd"]
    assert sum(agg["risk_counts"].values()) == 5


def test_score_portfolio_persists_runs():
    rec = _train_and_register("production")
    out = ps.score_portfolio(_portfolio(3), model_id=rec["id"])
    persisted = ps.list_runs()
    ids_in_response = {r["id"] for r in out["runs"]}
    ids_persisted = {r["id"] for r in persisted}
    assert ids_in_response.issubset(ids_persisted)


def test_score_portfolio_research_model_watermarked():
    rec = _train_and_register("research")
    out = ps.score_portfolio(_portfolio(3), model_id=rec["id"])
    assert out["watermark"] is not None
    assert all(r["watermark"] is not None for r in out["runs"])


def test_score_portfolio_attaches_segment_risks():
    rec = _train_and_register("production")
    upsert(
        "holdout_results",
        {
            "id": "vertical|travel",
            "segmentation_var": "vertical",
            "level": "travel",
            "within_r2_median": 0.70,
            "extrapolation_r2_median": 0.45,
            "penalty_pp": 25.0,
            "n_iterations": 20,
            "risk": "severe",
            "created_at": "2026-05-07T00:00:00Z",
        },
        key="id",
    )
    portfolio = _portfolio(5)
    portfolio[0]["vertical"] = "travel"
    out = ps.score_portfolio(portfolio, model_id=rec["id"])
    travel_run = next(r for r in out["runs"] if r["spec"]["campaign_id"] == "PORT-000")
    assert any(
        sr["risk"] == "severe" for sr in travel_run["segment_risks"]
    )
    assert out["worst_segment_risk"] is not None
    assert out["worst_segment_risk"]["risk"] == "severe"


def test_score_portfolio_empty_raises():
    _train_and_register("production")
    with pytest.raises(ps.PredictionError):
        ps.score_portfolio([])


def test_score_portfolio_no_models_raises():
    with pytest.raises(ps.PredictionError):
        ps.score_portfolio(_portfolio(3))


def test_score_portfolio_assigns_missing_campaign_ids():
    rec = _train_and_register("production")
    rows = _portfolio(3)
    for r in rows:
        del r["campaign_id"]
    out = ps.score_portfolio(rows, model_id=rec["id"])
    assert all(r["campaign_id"].startswith("PRED-") for r in out["runs"])
    assert len({r["campaign_id"] for r in out["runs"]}) == 3
