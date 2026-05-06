"""Tests for the mechanical-correlation defense (PDF §4.4)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from services.mechanical_correlation_defense import (
    MIN_USERS_FOR_AGGREGATED,
    decide,
    decide_for_dataframe,
    split_users,
)


def test_user_level_data_yields_sample_split():
    d = decide("CMP-001", has_user_level_data=True, test_users=2_000_000)
    assert d.mc_defense_mode == "sample_split"
    assert d.sample_split_seed is not None


def test_aggregated_above_threshold_yields_compromise():
    d = decide("CMP-002", has_user_level_data=False, test_users=5_000_000)
    assert d.mc_defense_mode == "shared_sample_compromise"
    assert d.sample_split_seed is None


def test_aggregated_below_threshold_blocks():
    d = decide(
        "CMP-003",
        has_user_level_data=False,
        test_users=MIN_USERS_FOR_AGGREGATED - 1,
    )
    assert d.mc_defense_mode == "blocked"
    assert d.sample_split_seed is None
    assert "research mode" in d.reason.lower()


def test_split_users_partitions_disjointly():
    s1, s2 = split_users(1000, "CMP-001")
    assert len(set(s1) & set(s2)) == 0
    assert len(s1) + len(s2) == 1000


def test_split_users_is_deterministic():
    s1a, s2a = split_users(1000, "CMP-001")
    s1b, s2b = split_users(1000, "CMP-001")
    np.testing.assert_array_equal(s1a, s1b)
    np.testing.assert_array_equal(s2a, s2b)


def test_split_users_differs_per_campaign():
    s1a, _ = split_users(1000, "CMP-001")
    s1b, _ = split_users(1000, "CMP-002")
    # Different seeds → different orderings (with overwhelming probability).
    assert not np.array_equal(s1a, s1b)


def test_split_users_rejects_tiny_input():
    with pytest.raises(ValueError):
        split_users(1, "CMP-001")


def test_decide_for_dataframe_respects_is_rct():
    df = pd.DataFrame(
        [
            {
                "campaign_id": "CMP-001",
                "is_rct": 1,
                "test_users": 2_000_000,
                "has_user_level_data": True,
            },
            {
                "campaign_id": "CMP-002",
                "is_rct": 0,
                "test_users": 0,
                "has_user_level_data": False,
            },
            {
                "campaign_id": "CMP-003",
                "is_rct": 1,
                "test_users": 500_000,
                "has_user_level_data": False,
            },
        ]
    )
    out = decide_for_dataframe(df)
    decisions = out["mc_defense"].tolist()
    assert decisions[0]["mc_defense_mode"] == "sample_split"
    assert decisions[1] is None  # non-RCT row
    assert decisions[2]["mc_defense_mode"] == "blocked"
