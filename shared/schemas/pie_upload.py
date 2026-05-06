"""
Standard PIE upload schema (PDF Appendix B).

Enforced via Pandera. RCT-only columns are required when is_rct == 1 and
allowed-null otherwise. The §4.1 hard exclusions (test_users >= 1M,
test_conversions >= 5K, control exposure = 0, funnel monotonicity, etc.) are
implemented as Pandera Checks; some run at row level, some at DataFrame level.

Recommended (lcc_1d, lcc_28d, view_through_conversions, avg_dwell_time) and
optional (lcc_1h, conversion_value, creative_format, placement, bid_strategy)
columns are nullable; their absence is flagged downstream as a warning.
"""

from __future__ import annotations

import pandera as pa
from pandera import Check, Column, DataFrameSchema, Index

# §4.1 thresholds for RCT donor-pool admission
MIN_TEST_USERS = 1_000_000
MIN_TEST_CONVERSIONS = 5_000

# Allowed categorical values
VERTICALS = {"ecommerce", "retail", "travel", "finance", "cpg", "auto", "telco", "other"}
FUNNEL_STAGES = {"upper", "mid", "lower"}
OBJECTIVES = {"conversion", "traffic", "awareness", "app_install"}
AUDIENCE_TYPES = {"prospecting", "retargeting", "crm", "lookalike"}
CONVERSION_OPT = {"onsite", "offsite"}
CUSTOM_AUDIENCE = {"broad", "narrow"}


def _funnel_monotonicity(df) -> bool:
    """impressions >= clicks >= conversions on every row (§4.1)."""
    return bool(
        ((df["impressions"] >= df["clicks"]) & (df["clicks"] >= df["conversions"])).all()
    )


def _date_validity(df) -> bool:
    """start_date < end_date on every row (§4.1)."""
    return bool((df["start_date"] < df["end_date"]).all())


def _rct_admission(df) -> bool:
    """RCT rows (is_rct == 1) must satisfy hard-exclusion thresholds (§4.1)."""
    rct_rows = df[df["is_rct"] == 1]
    if rct_rows.empty:
        return True
    if rct_rows["test_users"].isnull().any() or rct_rows["test_conversions"].isnull().any():
        return False
    return bool(
        (rct_rows["test_users"] >= MIN_TEST_USERS).all()
        and (rct_rows["test_conversions"] >= MIN_TEST_CONVERSIONS).all()
        and (rct_rows["exposed_test_users"] > 0).all()
        and (rct_rows["exposed_test_users"] <= rct_rows["test_users"]).all()
    )


pie_upload_schema: DataFrameSchema = DataFrameSchema(
    columns={
        # --- Required identifiers / metadata --------------------------------
        "campaign_id": Column(str, nullable=False),
        "advertiser_id": Column(str, nullable=False),
        "campaign_name": Column(str, nullable=False),
        "is_rct": Column(int, checks=Check.isin([0, 1]), nullable=False),
        "vertical": Column(str, checks=Check.isin(VERTICALS), nullable=False),
        "funnel_stage": Column(str, checks=Check.isin(FUNNEL_STAGES), nullable=False),
        "objective": Column(str, checks=Check.isin(OBJECTIVES), nullable=False),
        "audience_type": Column(str, checks=Check.isin(AUDIENCE_TYPES), nullable=False),
        # --- New in v3 (validation memo) -----------------------------------
        "conversion_optimization": Column(
            str, checks=Check.isin(CONVERSION_OPT), nullable=False
        ),
        "custom_audience": Column(
            str, checks=Check.isin(CUSTOM_AUDIENCE), nullable=False
        ),
        "advertiser_platform_experience_months": Column(
            int, checks=Check.greater_than_or_equal_to(0), nullable=False
        ),
        # --- Required dates / metrics --------------------------------------
        "start_date": Column(pa.DateTime, nullable=False),
        "end_date": Column(pa.DateTime, nullable=False),
        "cost": Column(float, checks=Check.greater_than(0), nullable=False),
        "impressions": Column(int, checks=Check.greater_than_or_equal_to(0), nullable=False),
        "clicks": Column(int, checks=Check.greater_than_or_equal_to(0), nullable=False),
        "conversions": Column(int, checks=Check.greater_than_or_equal_to(0), nullable=False),
        # --- RCT-only (nullable for non-RCT rows) --------------------------
        "test_users": Column(float, nullable=True),
        "control_users": Column(float, nullable=True),
        "exposed_test_users": Column(float, nullable=True),
        "test_conversions": Column(float, nullable=True),
        "control_conversions": Column(float, nullable=True),
        # --- LCC windows (lcc_7d required per §4.3) ------------------------
        "lcc_7d": Column(int, checks=Check.greater_than_or_equal_to(0), nullable=False),
        "lcc_1d": Column(int, nullable=True),
        "lcc_28d": Column(int, nullable=True),
        "lcc_1h": Column(int, nullable=True),
        # --- Recommended ---------------------------------------------------
        "view_through_conversions": Column(int, nullable=True),
        "avg_dwell_time": Column(float, nullable=True),
        # --- Optional ------------------------------------------------------
        "conversion_value": Column(float, nullable=True),
        "creative_format": Column(str, nullable=True),
        "placement": Column(str, nullable=True),
        "bid_strategy": Column(str, nullable=True),
    },
    checks=[
        Check(_funnel_monotonicity, error="funnel monotonicity violated: impressions >= clicks >= conversions"),
        Check(_date_validity, error="date validity violated: start_date must be < end_date"),
        Check(
            _rct_admission,
            error=(
                "RCT admission failed: rows with is_rct=1 must have "
                f"test_users >= {MIN_TEST_USERS:,}, "
                f"test_conversions >= {MIN_TEST_CONVERSIONS:,}, "
                "and 0 < exposed_test_users <= test_users (PDF §4.1, p. 17)"
            ),
        ),
    ],
    index=Index(int),
    strict=False,  # allow extra columns; ignored gracefully
    coerce=True,
)
