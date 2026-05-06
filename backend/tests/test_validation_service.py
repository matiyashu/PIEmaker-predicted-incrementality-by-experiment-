"""Tests for the validation engine (PDF §4.1, §4.3)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from services.validation_service import validate_dataframe

SAMPLE_CSV = Path(__file__).resolve().parents[2] / "shared" / "examples" / "sample_rct_upload.csv"


def _load() -> pd.DataFrame:
    return pd.read_csv(SAMPLE_CSV, parse_dates=["start_date", "end_date"])


def _rule(result, rule_id: str) -> dict:
    for r in result.rules:
        if r["rule_id"] == rule_id:
            return r
    raise AssertionError(f"rule {rule_id} not found in result")


def test_clean_sample_passes_critical_rules():
    df = _load()
    result = validate_dataframe(df)
    assert result.block_training is False
    assert result.data_quality_score >= 90
    assert _rule(result, "rct_test_users_min")["passed"] is True
    assert _rule(result, "rct_test_conversions_min")["passed"] is True


def test_under_powered_rct_blocks_training():
    df = _load()
    df.loc[0, "test_users"] = 500_000  # below 1M
    result = validate_dataframe(df)
    assert result.block_training is True
    rule = _rule(result, "rct_test_users_min")
    assert rule["passed"] is False
    assert rule["severity"] == "critical"
    assert 0 in rule["affected_rows"]


def test_low_conversion_rct_blocks_training():
    df = _load()
    df.loc[0, "test_conversions"] = 1_000  # below 5K
    result = validate_dataframe(df)
    assert result.block_training is True
    rule = _rule(result, "rct_test_conversions_min")
    assert rule["passed"] is False


def test_zero_cost_blocks_training():
    df = _load()
    df.loc[2, "cost"] = 0
    result = validate_dataframe(df)
    assert result.block_training is True
    rule = _rule(result, "cost_positive")
    assert rule["passed"] is False
    assert 2 in rule["affected_rows"]


def test_funnel_violation_blocks_training():
    df = _load()
    df.loc[0, "clicks"] = df.loc[0, "impressions"] + 1
    result = validate_dataframe(df)
    assert result.block_training is True
    rule = _rule(result, "funnel_monotonicity")
    assert rule["passed"] is False


def test_missing_lcc_7d_critical():
    df = _load()
    df.loc[0, "lcc_7d"] = None
    result = validate_dataframe(df)
    rule = _rule(result, "lcc_7d_required")
    assert rule["passed"] is False
    assert rule["severity"] == "critical"
    assert result.block_training is True


def test_missing_lcc_1d_warning_overridable():
    df = _load()
    df["lcc_1d"] = None
    result = validate_dataframe(df)
    rule = _rule(result, "lcc_1d_recommended")
    assert rule["passed"] is False
    assert rule["severity"] == "warning"


def test_data_quality_score_decreases_with_warnings():
    clean = validate_dataframe(_load())
    df = _load()
    df["lcc_28d"] = None
    warned = validate_dataframe(df)
    assert warned.data_quality_score <= clean.data_quality_score


def test_severity_breakdown_reports_counts():
    df = _load()
    df.loc[0, "test_users"] = 500_000
    df["lcc_28d"] = None
    result = validate_dataframe(df)
    assert result.severity_breakdown["critical"] >= 1
    assert result.severity_breakdown["warning"] >= 1


def test_rules_include_paper_references():
    df = _load()
    df.loc[0, "test_users"] = 500_000
    result = validate_dataframe(df)
    rule = _rule(result, "rct_test_users_min")
    assert "p. 17" in rule["paper_reference"]


def test_date_validity_rule():
    df = _load()
    df.loc[0, "end_date"] = df.loc[0, "start_date"]
    result = validate_dataframe(df)
    rule = _rule(result, "date_validity")
    assert rule["passed"] is False
