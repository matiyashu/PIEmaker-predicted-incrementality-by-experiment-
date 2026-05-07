"""Decision Simulator tests (Prompt 4.2 — final)."""

from __future__ import annotations

import random

import pytest

from ml import model_registry
from ml.train_random_forest import train
from services import simulator_service as ss
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
    rng = random.Random(404)
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
        name="sim-test-rf",
        algorithm="random_forest",
        feature_set_version="v1",
        hyperparameters=artifacts.chosen_hyperparameters,
        training_donor_pool_size=400 if status == "production" else 250,
        estimator=artifacts.estimator,
        status=status,
    )
    model_registry.record_metric(
        rec["id"], "weighted_r_squared_bootstrap_mean",
        0.78, ci_lower=0.72, ci_upper=0.84,
    )
    return rec


def _portfolio(n: int = 5, *, with_severe_segment: bool = False) -> list[dict]:
    rng = random.Random(505)
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
            "advertiser_platform_experience_months": 18,
            "creative_format": "video",
            "placement": "feed",
            "bid_strategy": "lowest_cost",
            "market": "US",
            "spend_tier": "high",
            "platform": "meta",
            "start_date": "2026-06-01",
            "end_date": "2026-06-22",
            "cost": 50_000.0 + i * 10_000.0,
            "test_users": 1_500_000,
            "exposed_test_users": 1_200_000,
            "clicks": 30_000,
            "impressions": 4_000_000,
            "conversions": 28_000,
            "lcc_7d": 5_000,
        })
    if with_severe_segment:
        # Persist a holdout that flags 'media' as severe; tag campaign 0
        upsert(
            "holdout_results",
            {
                "id": "vertical|media",
                "segmentation_var": "vertical",
                "level": "media",
                "within_r2_median": 0.65,
                "extrapolation_r2_median": 0.30,
                "penalty_pp": 35.0,
                "n_iterations": 20,
                "risk": "severe",
                "created_at": "2026-05-07T00:00:00Z",
            },
            key="id",
        )
        out[0]["vertical"] = "media"
    return out


def test_simulator_runs_on_production_model():
    rec = _train_and_register("production")
    out = ss.simulate(_portfolio(5), model_id=rec["id"])
    assert out["model"]["id"] == rec["id"]
    assert out["n_campaigns"] == 5
    assert out["original_total_budget"] == sum(50_000 + i * 10_000 for i in range(5))


def test_simulator_blocks_research_model():
    rec = _train_and_register("research")
    with pytest.raises(ss.SimulatorError) as exc:
        ss.simulate(_portfolio(3), model_id=rec["id"])
    assert "production" in str(exc.value).lower()


def test_simulator_no_models_raises():
    with pytest.raises(ss.SimulatorError):
        ss.simulate(_portfolio(3))


def test_simulator_empty_rows_raises():
    _train_and_register("production")
    with pytest.raises(ss.SimulatorError):
        ss.simulate([])


def test_budget_conserved_within_total():
    """Sum of proposed spends must equal total_budget (within float tolerance)."""
    rec = _train_and_register("production")
    out = ss.simulate(_portfolio(5), model_id=rec["id"], cap_multiplier=10.0)
    proposed = sum(a["proposed_spend"] for a in out["allocations"])
    assert proposed == pytest.approx(out["total_budget"], abs=1e-2)


def test_cap_multiplier_enforced():
    """No campaign's proposed spend should exceed original × cap_multiplier."""
    rec = _train_and_register("production")
    out = ss.simulate(_portfolio(5), model_id=rec["id"], cap_multiplier=1.5)
    for a in out["allocations"]:
        if a["original_spend"] > 0:
            assert a["proposed_spend"] <= a["original_spend"] * 1.5 + 1e-6


def test_blocked_campaign_gets_zero_proposed_spend():
    rec = _train_and_register("production")
    rows = _portfolio(5, with_severe_segment=True)
    out = ss.simulate(rows, model_id=rec["id"], cap_multiplier=2.0)
    blocked = [a for a in out["allocations"] if a["action"] == "block"]
    assert blocked  # at least one was blocked
    for a in blocked:
        assert a["proposed_spend"] == 0.0


def test_total_budget_override_used_when_provided():
    rec = _train_and_register("production")
    out = ss.simulate(
        _portfolio(5),
        model_id=rec["id"],
        total_budget_override=1_000_000.0,
        cap_multiplier=10.0,
    )
    assert out["total_budget"] == 1_000_000.0
    proposed_sum = sum(a["proposed_spend"] for a in out["allocations"])
    assert proposed_sum == pytest.approx(1_000_000.0, abs=1.0)


def test_ic_math_consistent():
    """proposed_ic == predicted_icpd × proposed_spend per row, and the totals match."""
    rec = _train_and_register("production")
    out = ss.simulate(_portfolio(5), model_id=rec["id"])
    for a in out["allocations"]:
        assert a["proposed_ic"] == pytest.approx(
            a["predicted_icpd"] * a["proposed_spend"], abs=1e-3
        )
    total = sum(a["proposed_ic"] for a in out["allocations"])
    assert total == pytest.approx(out["proposed_ic_total"], abs=1e-2)


def test_simulator_preserves_original_when_caps_force_status_quo():
    """With cap_multiplier=1.0, no campaign can grow → proposed totals can't
    exceed original budget."""
    rec = _train_and_register("production")
    rows = _portfolio(5)
    out = ss.simulate(rows, model_id=rec["id"], cap_multiplier=1.0)
    for a in out["allocations"]:
        assert a["proposed_spend"] <= a["original_spend"] + 1e-6
