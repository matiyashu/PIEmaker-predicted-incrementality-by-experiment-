"""Feature engineering studio tests (Prompt 2.2)."""

from __future__ import annotations

import pandas as pd
import pytest

from services.feature_engineering_service import (
    DEFAULT_FEATURE_SET_VERSION,
    FEATURE_TABLE,
    X_POST_FIELDS,
    X_PRE_FIELDS,
    build_features,
)
from services.persistence import read_table, reset


@pytest.fixture(autouse=True)
def _reset_state():
    reset()
    yield
    reset()


def _campaign_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "campaign_id": "CMP-001",
                "objective": "conversions",
                "vertical": "ecommerce",
                "audience_type": "retargeting",
                "funnel_stage": "lower",
                "conversion_optimization": "yes",
                "custom_audience": "yes",
                "advertiser_platform_experience_months": 24,
                "creative_format": "video",
                "placement": "feed",
                "bid_strategy": "lowest_cost",
                "market": "US",
                "spend_tier": "high",
                "platform": "meta",
                "start_date": "2026-01-15",
                "end_date": "2026-02-05",
                "cost": 50_000.0,
                "exposed_test_users": 1_600_000,
                "test_users": 2_000_000,
                "clicks": 100_000,
                "impressions": 5_000_000,
                "conversions": 50_000,
                "lcc_1h": 800,
                "lcc_1d": 1_200,
                "lcc_7d": 2_400,
                "lcc_28d": 4_000,
                "view_through_conversions": 600,
                "avg_dwell_time": 12.5,
            },
        ]
    )


def test_v3_features_present_in_x_pre():
    for f in (
        "conversion_optimization",
        "custom_audience",
        "advertiser_platform_experience_months",
    ):
        assert f in X_PRE_FIELDS


def test_build_features_training_mode_persists_row():
    df = build_features(_campaign_frame(), mode="training")
    assert len(df) == 1
    row = df.iloc[0]
    assert row["mode"] == "training"
    assert row["feature_set_version"] == DEFAULT_FEATURE_SET_VERSION
    assert row["x_pre"]["objective"] == "conversions"
    assert row["x_pre"]["month"] == 1
    assert row["x_pre"]["quarter"] == 1
    assert row["x_pre"]["campaign_duration_days"] == 21
    persisted = read_table(FEATURE_TABLE)
    assert len(persisted) == 1
    assert persisted[0]["campaign_id"] == "CMP-001"


def test_build_features_x_post_ratios():
    df = build_features(_campaign_frame(), mode="training")
    x_post = df.iloc[0]["x_post"]
    assert x_post["exposure_rate"] == pytest.approx(0.8)
    assert x_post["ctr"] == pytest.approx(0.02)
    assert x_post["clicks_per_dollar"] == pytest.approx(2.0)
    assert x_post["conversions_per_dollar"] == pytest.approx(1.0)
    assert x_post["lcc_7d_per_dollar"] == pytest.approx(0.048)
    assert x_post["avg_dwell_time"] == pytest.approx(12.5)


def test_build_features_scoring_mode_tagged():
    df = build_features(_campaign_frame(), mode="scoring", sample_id="sample-2")
    assert df.iloc[0]["mode"] == "scoring"
    assert df.iloc[0]["sample_id"] == "sample-2"


def test_build_features_invalid_mode_raises():
    with pytest.raises(ValueError):
        build_features(_campaign_frame(), mode="prediction")  # type: ignore[arg-type]


def test_build_features_missing_columns_yield_none():
    minimal = pd.DataFrame(
        [{"campaign_id": "CMP-002"}]  # no fields beyond ID
    )
    df = build_features(minimal, mode="training")
    row = df.iloc[0]
    assert row["x_pre"]["objective"] is None
    assert row["x_post"]["exposure_rate"] is None


def test_x_post_fields_constant_matches_keys():
    df = build_features(_campaign_frame(), mode="training")
    keys = set(df.iloc[0]["x_post"].keys())
    assert keys == set(X_POST_FIELDS)
