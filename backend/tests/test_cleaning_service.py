"""Tests for the Cleaning Workbench (PDF §4.5)."""

from __future__ import annotations

import pandas as pd

from services.cleaning_service import (
    dedupe,
    handle_missing_values,
    hard_block_invalid_metrics,
    normalize_categoricals,
    normalize_currency_to_usd,
    normalize_dates_iso,
    run_pipeline,
    standardize_campaign_ids,
    winsorize,
)


def _make_minimal_df() -> pd.DataFrame:
    """Minimal valid frame covering the 17 critical fields."""
    return pd.DataFrame([
        {
            "campaign_id": "  cmp-001 ",
            "advertiser_id": "ADV-100",
            "is_rct": 1,
            "vertical": "ecommerce",
            "funnel_stage": "lower",
            "objective": "conversion",
            "audience_type": "RTG",
            "conversion_optimization": "onsite",
            "custom_audience": "narrow",
            "advertiser_platform_experience_months": 18,
            "start_date": "2026-01-01",
            "end_date": "2026-01-31",
            "cost": 50000.0,
            "impressions": 1_000_000,
            "clicks": 50_000,
            "conversions": 5_000,
            "lcc_7d": 12_000,
            "lcc_1d": 4_000,
            "lcc_28d": 14_500,
        },
        {
            "campaign_id": "Cmp_002@!",
            "advertiser_id": "ADV-100",
            "is_rct": 0,
            "vertical": "ecommerce",
            "funnel_stage": "lower",
            "objective": "conversion",
            "audience_type": "remarketing",
            "conversion_optimization": "onsite",
            "custom_audience": "narrow",
            "advertiser_platform_experience_months": 18,
            "start_date": "2026-02-01",
            "end_date": "2026-02-28",
            "cost": 30000.0,
            "impressions": 600_000,
            "clicks": 30_000,
            "conversions": 3_000,
            "lcc_7d": 8_000,
            "lcc_1d": None,
            "lcc_28d": 9_200,
        },
    ])


def test_standardize_campaign_ids_strips_and_uppercases():
    df = _make_minimal_df()
    out, action = standardize_campaign_ids(df)
    assert out.loc[0, "campaign_id"] == "CMP-001"
    assert out.loc[1, "campaign_id"] == "CMP_002"
    assert action.action_type == "standardize_campaign_ids"
    assert action.rows_affected == 2


def test_standardize_idempotent():
    df = _make_minimal_df()
    once, _ = standardize_campaign_ids(df)
    twice, action = standardize_campaign_ids(once)
    assert action.rows_affected == 0
    assert (once["campaign_id"] == twice["campaign_id"]).all()


def test_normalize_categoricals_maps_audience_aliases():
    df = _make_minimal_df()
    out, action = normalize_categoricals(df)
    assert (out["audience_type"] == "retargeting").all()
    assert action.rows_affected == 2


def test_normalize_dates_iso_handles_mixed_formats():
    df = _make_minimal_df()
    df.loc[0, "start_date"] = "01/15/2026"  # non-ISO
    out, action = normalize_dates_iso(df)
    assert pd.api.types.is_datetime64_any_dtype(out["start_date"])
    assert action.action_type == "normalize_dates_iso"


def test_normalize_currency_to_usd_no_op_without_currency_column():
    df = _make_minimal_df()
    out, action = normalize_currency_to_usd(df, fx_rates={"IDR": 0.000061})
    assert action.rows_affected == 0
    assert "cost_original" not in out.columns


def test_normalize_currency_to_usd_converts_and_preserves_original():
    df = _make_minimal_df()
    df["currency_code"] = ["IDR", "USD"]
    df.loc[0, "cost"] = 1_000_000_000.0  # IDR
    out, action = normalize_currency_to_usd(df, fx_rates={"IDR": 0.000061})
    assert out.loc[0, "cost_original"] == 1_000_000_000.0
    assert out.loc[0, "cost"] == 61_000.0
    assert action.rows_affected == 1


def test_handle_missing_values_rejects_critical_nulls():
    df = _make_minimal_df()
    df.loc[0, "cost"] = None
    out, action = handle_missing_values(df)
    assert len(out) == 1
    assert action.before_summary["rows_rejected_critical_null"] == 1


def test_handle_missing_values_imputes_recommended_nulls():
    df = _make_minimal_df()
    out, action = handle_missing_values(df)
    assert out["lcc_1d"].notnull().all()
    assert action.before_summary["rows_imputed_recommended"] >= 1


def test_dedupe_keeps_latest():
    df = _make_minimal_df()
    duplicate = df.iloc[[0]].copy()
    duplicate.loc[:, "cost"] = 99_999.0
    df = pd.concat([df, duplicate], ignore_index=True)
    out, action = dedupe(df)
    assert action.rows_affected == 1
    assert (out["campaign_id"].value_counts() == 1).all()


def test_winsorize_disabled_by_default():
    df = _make_minimal_df()
    out, action = winsorize(df)
    assert action.rows_affected == 0
    assert "Winsorization disabled" in (action.notes or "")


def test_winsorize_clips_when_enabled():
    df = pd.DataFrame({"icpd": [-100, 0.1, 0.2, 0.3, 1000]})
    out, action = winsorize(df, columns=["icpd"], enabled=True)
    assert out["icpd"].min() > -100
    assert out["icpd"].max() < 1000
    assert action.rows_affected >= 1


def test_hard_block_invalid_metrics_drops_violations():
    df = _make_minimal_df()
    df.loc[0, "clicks"] = df.loc[0, "impressions"] + 1  # violates funnel
    out, action = hard_block_invalid_metrics(df)
    assert len(out) == 1
    assert action.rows_affected == 1


def test_pipeline_runs_all_eight_actions():
    df = _make_minimal_df()
    out, actions = run_pipeline(df, applied_by="prima")
    types = {a.action_type for a in actions}
    expected = {
        "standardize_campaign_ids",
        "normalize_categoricals",
        "normalize_dates_iso",
        "normalize_currency_to_usd",
        "handle_missing_values",
        "dedupe",
        "winsorize",
        "hard_block_invalid_metrics",
    }
    assert expected.issubset(types)
    assert all(a.applied_by == "prima" for a in actions)
    assert len(out) <= len(df)
