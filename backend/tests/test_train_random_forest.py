"""Random Forest training pipeline tests (Prompt 2.3)."""

from __future__ import annotations

import random

import pytest

from ml.train_random_forest import predict_icpd, train


def _rows(n: int = 30) -> tuple[list[dict], list[dict]]:
    """Synthesize a small donor pool with linear ICPD signal."""
    rng = random.Random(7)
    feats: list[dict] = []
    labels: list[dict] = []
    for i in range(n):
        cid = f"CMP-{i:03d}"
        vertical = rng.choice(["ecommerce", "travel", "finance"])
        cpd = rng.uniform(0.5, 5.0)
        lcc7 = rng.uniform(0.05, 0.5)
        # Engineered ICPD = 0.3 * conversions_per_dollar + 1.2 * lcc_7d_per_dollar + noise
        icpd_val = 0.3 * cpd + 1.2 * lcc7 + rng.gauss(0, 0.02)
        feats.append(
            {
                "campaign_id": cid,
                "feature_set_version": "v1",
                "mode": "training",
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


def test_train_returns_artifacts_with_chosen_params():
    feats, labels = _rows(30)
    artifacts = train(feats, labels, n_splits=3)
    assert artifacts.n_observations == 30
    assert "n_estimators" in artifacts.chosen_hyperparameters
    assert len(artifacts.cv_scores) == 3
    assert artifacts.feature_columns  # non-empty


def test_predict_icpd_returns_one_value_per_row():
    feats, labels = _rows(20)
    artifacts = train(feats, labels, n_splits=3)
    preds = predict_icpd(artifacts.estimator, feats[:5])
    assert len(preds) == 5


def test_train_rejects_empty_inputs():
    with pytest.raises(ValueError):
        train([], [])


def test_train_rejects_too_few_aligned_rows():
    feats, labels = _rows(10)
    # Misalign so only 2 rows match
    labels = labels[:2]
    with pytest.raises(ValueError):
        train(feats, labels, n_splits=2)


def test_train_with_explicit_grid_picks_from_it():
    feats, labels = _rows(20)
    grid = [{"n_estimators": 50, "max_depth": 5, "min_samples_leaf": 2,
             "max_features": "sqrt"}]
    artifacts = train(feats, labels, grid=grid, n_splits=3)
    assert artifacts.chosen_hyperparameters == grid[0]
