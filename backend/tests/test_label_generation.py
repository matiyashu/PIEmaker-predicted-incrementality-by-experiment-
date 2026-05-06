"""Label generator tests (Prompt 2.2)."""

from __future__ import annotations

import pandas as pd
import pytest

from services.label_generation_service import generate_labels
from services.persistence import read_table, reset


@pytest.fixture(autouse=True)
def _reset_state():
    reset()
    yield
    reset()


def _rct_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "campaign_id": "CMP-001",
                "test_users": 2_000_000,
                "control_users": 2_000_000,
                "exposed_test_users": 1_600_000,
                "test_conversions": 50_000,
                "control_conversions": 40_000,
                "cost": 50_000.0,
            },
            {
                "campaign_id": "CMP-002",
                "test_users": 1_500_000,
                "control_users": 1_500_000,
                "exposed_test_users": 1_200_000,
                "test_conversions": 30_000,
                "control_conversions": 24_000,
                "cost": 75_000.0,
            },
        ]
    )


def test_labels_match_frozen_formula_output():
    df = _rct_frame()
    out = generate_labels(df, has_user_level_data=False)
    # Hand-computed for row 0:
    # D̄ = 1.6M/2M = 0.8
    # Y_tr = 50K/2M = 0.025
    # Y_cr = 40K/2M = 0.020
    # ATT = (0.025 - 0.020) / 0.8 = 0.00625
    # IC  = 0.00625 * 0.8 * 2M = 10000
    # ICPD = 10000 / 50000 = 0.2
    assert out.loc[0, "att"] == pytest.approx(0.00625)
    assert out.loc[0, "incremental_conversions"] == pytest.approx(10_000)
    assert out.loc[0, "icpd"] == pytest.approx(0.2)


def test_mc_defense_mode_recorded():
    out = generate_labels(_rct_frame(), has_user_level_data=False)
    assert (out["mc_defense_mode"] == "shared_sample_compromise").all()
    out2 = generate_labels(_rct_frame(), has_user_level_data=True)
    assert (out2["mc_defense_mode"] == "sample_split").all()
    assert out2["sample_split_seed"].notnull().all()


def test_labels_persisted_to_state():
    generate_labels(_rct_frame(), has_user_level_data=False)
    rows = read_table("rct_labels")
    assert {r["campaign_id"] for r in rows} == {"CMP-001", "CMP-002"}
