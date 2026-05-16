"""
Feature Engineering Studio (Prompt 2.2 + V.4 Wave 1).

Builds X_pre and X_post features for training and scoring. Persists every
row under a composite key (campaign_id, feature_set_version, mode) so a
scoring pass over the same campaign no longer silently overwrites the
training row — that overwrite is what caused several V.3 race conditions.

V.4 also adds three paper-aligned X_pre fields:
  * advertiser_id    — needed for existing-vs-new advertiser CV (§5.3)
  * advertiser_size  — controls for advertiser scale (paper feature set)
  * campaign_year    — year-to-year drift signal (Table 1: 21pp R² penalty)

And persists the true campaign `cost` alongside each row so paper-mode
evaluation can use it as the weighted-R² weight (replacing the V.3 proxy
`1/conversions_per_dollar`).
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
    "conversion_optimization",       # v3
    "custom_audience",                # v3
    "advertiser_platform_experience_months",  # v3
    "advertiser_id",                  # V.4 NEW — required for advertiser-cohort CV
    "advertiser_size",                # V.4 NEW — paper feature set, controls for scale
    "campaign_year",                  # V.4 NEW — year-to-year drift signal
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


def composite_key(
    campaign_id: str, feature_set_version: str, mode: str
) -> str:
    """Composite feature_store key (V.4): scoping rows by (campaign, version, mode).

    Used as the upsert key to prevent a scoring pass from overwriting the
    matching training row for the same campaign.
    """
    return f"{campaign_id}|{feature_set_version}|{mode}"

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
    out["advertiser_id"] = df.get("advertiser_id")
    out["advertiser_size"] = df.get("advertiser_size")
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
        out["campaign_year"] = sd.dt.year
    else:
        out["month"] = None
        out["quarter"] = None
        out["campaign_year"] = df.get("campaign_year")
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
        # V.4: persist true campaign cost alongside the row so paper-mode
        # evaluation can use it as the weighted-R² weight.
        raw_cost = df.loc[idx, "cost"] if "cost" in df.columns else None
        cost_value: float | None
        if raw_cost is None or pd.isnull(raw_cost):
            cost_value = None
        else:
            try:
                cost_value = float(raw_cost)
            except (TypeError, ValueError):
                cost_value = None

        row = {
            # V.4 composite key — see composite_key() above. Keeps
            # campaign_id as its own field for legacy readers.
            "id": composite_key(cid, feature_set_version, mode),
            "campaign_id": cid,
            "feature_set_version": feature_set_version,
            "mode": mode,
            "sample_id": sample_id,
            "cost": cost_value,
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
        # V.4 composite key — was previously campaign_id, which caused
        # scoring rows to silently overwrite training rows.
        upsert(FEATURE_TABLE, row, key="id")
        rows.append(row)
    return pd.DataFrame(rows)
