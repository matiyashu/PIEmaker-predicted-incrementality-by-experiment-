"""
Pandera schema tests for the standard PIE upload format (PDF Appendix B + §4.1).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pandera as pa
import pytest

# Make the /shared directory importable.
SHARED_DIR = Path(__file__).resolve().parents[2] / "shared"
import sys

sys.path.insert(0, str(SHARED_DIR))

from schemas.pie_upload import pie_upload_schema  # noqa: E402

SAMPLE_CSV = SHARED_DIR / "examples" / "sample_rct_upload.csv"


def _load_sample() -> pd.DataFrame:
    return pd.read_csv(SAMPLE_CSV, parse_dates=["start_date", "end_date"])


def test_sample_csv_validates():
    df = _load_sample()
    pie_upload_schema.validate(df, lazy=True)


def test_funnel_monotonicity_violation_rejected():
    df = _load_sample()
    df.loc[0, "clicks"] = df.loc[0, "impressions"] + 1  # clicks > impressions
    with pytest.raises(pa.errors.SchemaErrors):
        pie_upload_schema.validate(df, lazy=True)


def test_zero_cost_rejected():
    df = _load_sample()
    df.loc[0, "cost"] = 0
    with pytest.raises(pa.errors.SchemaErrors):
        pie_upload_schema.validate(df, lazy=True)


def test_rct_below_test_users_threshold_rejected():
    df = _load_sample()
    df.loc[0, "test_users"] = 500_000  # below 1M
    with pytest.raises(pa.errors.SchemaErrors):
        pie_upload_schema.validate(df, lazy=True)


def test_rct_below_test_conversions_threshold_rejected():
    df = _load_sample()
    df.loc[0, "test_conversions"] = 1_000  # below 5K
    with pytest.raises(pa.errors.SchemaErrors):
        pie_upload_schema.validate(df, lazy=True)


def test_invalid_vertical_rejected():
    df = _load_sample()
    df.loc[0, "vertical"] = "not_a_vertical"
    with pytest.raises(pa.errors.SchemaErrors):
        pie_upload_schema.validate(df, lazy=True)


def test_date_validity_rejected():
    df = _load_sample()
    df.loc[0, "end_date"] = df.loc[0, "start_date"]
    with pytest.raises(pa.errors.SchemaErrors):
        pie_upload_schema.validate(df, lazy=True)
