"""
Feature Engineering Studio (Prompt 2.2).

Builds X_pre and X_post features for training and scoring. Includes the three
new v3 features (conversion_optimization, custom_audience,
advertiser_platform_experience) per the validation memo.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

import pandas as pd

from services.persistence import upsert

FEATURE_TABLE = "feature_store"
DEFAULT_FEATURE_SET_VERSION = "v1"


# Pre-determined features (X_pre) — fixed before launch
X_PRE_FIELDS: list[str] = [
    "objective",
    "vertical",
    "audience_type",
    "funnel_stage",
    "conversion_optimization",       # v3 NEW
    "custom_audience",                # v3 NEW
    "advertiser_platform_experience_months",  # v3 NEW
    "creative_format",
    "placement",
    "bid_strategy",
    "market",
    "spend_tier",
    "platform",
    "month",
    "quarter",
    "campaign_duration_days",
]

# Post-determined features (X_post) — only knowable after the campaign
X_POST_FIELDS: list[str] = [
    "exposure_rate",
    "ctr",
    "clicks_per_dollar",
    "conversions_per_dollar",
    "lcc_1h_per_dollar",
    "lcc_1d_per_dollar",
    "lcc_7d_per_dollar",
    "lcc_28d_per_dollar",
    "view_through_per_dollar",
    "avg_dwell_time",
]

Mode = Literal["training", "scoring"]


def _safe_div(num, den):
    return (num / den).where(den > 0)


def _build_x_pre(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["objective"] = df.get("objective")
    out["vertical"] = df.get("vertical")
    out["audience_type"] = df.get("audience_type")
    out["funnel_stage"] = df.get("funnel_stage")
    out["conversion_optimization"] = df.get("conversion_optimization")
    out["custom_audience"] = df.get("custom_audience")
    out["advertiser_platform_experience_months"] = df.get(
        "advertiser_platform_experience_months"
    )
    out["creative_format"] = df.get("creative_format")
    out["placement"] = df.get("placement")
    out["bid_strategy"] = df.get("bid_strategy")
    out["market"] = df.get("market")
    out["spend_tier"] = df.get("spend_tier")
    out["platform"] = df.get("platform")

    if "start_date" in df.columns:
        sd = pd.to_datetime(df["start_date"], errors="coerce")
        out["month"] = sd.dt.month
        out["quarter"] = sd.dt.quarter
    else:
        out["month"] = None
        out["quarter"] = None
    if "start_date" in df.columns and "end_date" in df.columns:
        sd = pd.to_datetime(df["start_date"], errors="coerce")
        ed = pd.to_datetime(df["end_date"], errors="coerce")
        out["campaign_duration_days"] = (ed - sd).dt.days
    else:
        out["campaign_duration_days"] = None
    return out


def _build_x_post(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    cost = df["cost"].astype(float) if "cost" in df.columns else None
    if "exposed_test_users" in df.columns and "test_users" in df.columns:
        out["exposure_rate"] = _safe_div(
            df["exposed_test_users"].astype(float),
            df["test_users"].astype(float),
        )
    else:
        out["exposure_rate"] = None
    if "clicks" in df.columns and "impressions" in df.columns:
        out["ctr"] = _safe_div(df["clicks"].astype(float), df["impressions"].astype(float))
    else:
        out["ctr"] = None
    if cost is not None and "clicks" in df.columns:
        out["clicks_per_dollar"] = _safe_div(df["clicks"].astype(float), cost)
    if cost is not None and "conversions" in df.columns:
        out["conversions_per_dollar"] = _safe_div(df["conversions"].astype(float), cost)
    for col, target in (
        ("lcc_1h", "lcc_1h_per_dollar"),
        ("lcc_1d", "lcc_1d_per_dollar"),
        ("lcc_7d", "lcc_7d_per_dollar"),
        ("lcc_28d", "lcc_28d_per_dollar"),
        ("view_through_conversions", "view_through_per_dollar"),
    ):
        if cost is not None and col in df.columns:
            out[target] = _safe_div(df[col].astype(float), cost)
        else:
            out[target] = None
    out["avg_dwell_time"] = df.get("avg_dwell_time")
    return out


def build_features(
    df: pd.DataFrame,
    mode: Mode = "training",
    feature_set_version: str = DEFAULT_FEATURE_SET_VERSION,
    sample_id: str | None = None,
) -> pd.DataFrame:
    """Build X_pre + X_post for either training or scoring.

    For training mode with sample_split active, X_post should be computed from
    Sample 2 only (caller passes the Sample-2 slice). The function itself does
    not split — it just records the sample_id tag for provenance.
    """
    if mode not in ("training", "scoring"):
        raise ValueError("mode must be 'training' or 'scoring'")
    x_pre = _build_x_pre(df)
    x_post = _build_x_post(df)

    rows = []
    for idx in df.index:
        cid = str(df.loc[idx, "campaign_id"])
        row = {
            "campaign_id": cid,
            "feature_set_version": feature_set_version,
            "mode": mode,
            "sample_id": sample_id,
            "x_pre": {k: x_pre.loc[idx, k] for k in x_pre.columns},
            "x_post": {k: x_post.loc[idx, k] for k in x_post.columns},
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        # JSON-safe scalar coercion
        row["x_pre"] = {
            k: (None if pd.isnull(v) else (v.item() if hasattr(v, "item") else v))
            for k, v in row["x_pre"].items()
        }
        row["x_post"] = {
            k: (None if pd.isnull(v) else (v.item() if hasattr(v, "item") else v))
            for k, v in row["x_post"].items()
        }
        upsert(
            FEATURE_TABLE,
            row,
            key="campaign_id",  # one row per (campaign, version, mode) — prod
        )
        rows.append(row)
    return pd.DataFrame(rows)
