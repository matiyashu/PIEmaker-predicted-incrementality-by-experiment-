"""
Cleaning Workbench (PDF §4.5, Prompt 1.2).

Eight cleaning rules. Each function returns (cleaned_df, CleaningAction). The
CleaningAction record is persisted to the cleaning_actions table for audit.

Rules:
  1. standardize_campaign_ids
  2. normalize_categoricals
  3. normalize_dates_iso
  4. normalize_currency_to_usd
  5. handle_missing_values
  6. dedupe
  7. winsorize (optional, off by default)
  8. hard_block_invalid_metrics
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class CleaningAction:
    action_type: str
    rows_affected: int
    before_summary: dict = field(default_factory=dict)
    after_summary: dict = field(default_factory=dict)
    applied_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    applied_by: Optional[str] = None
    notes: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


# --- Rule 1: standardize campaign IDs ---------------------------------------


_ID_KEEP = re.compile(r"[^A-Z0-9_\-]")


def standardize_campaign_ids(
    df: pd.DataFrame, applied_by: str | None = None
) -> tuple[pd.DataFrame, CleaningAction]:
    """Trim, uppercase, strip non-alphanumerics except dash and underscore."""
    out = df.copy()
    before = out["campaign_id"].astype(str).tolist()
    cleaned = [_ID_KEEP.sub("", v.strip().upper()) for v in before]
    out["campaign_id"] = cleaned
    affected = sum(1 for a, b in zip(before, cleaned) if a != b)
    return out, CleaningAction(
        action_type="standardize_campaign_ids",
        rows_affected=affected,
        before_summary={"sample": before[:5]},
        after_summary={"sample": cleaned[:5]},
        applied_by=applied_by,
    )


# --- Rule 2: normalize categorical fields -----------------------------------


_AUDIENCE_MAP = {
    "rtg": "retargeting",
    "remarketing": "retargeting",
    "rmkt": "retargeting",
    "retarget": "retargeting",
    "retargeting": "retargeting",
    "prospecting": "prospecting",
    "prosp": "prospecting",
    "new_user": "prospecting",
    "newuser": "prospecting",
    "lookalike": "lookalike",
    "lal": "lookalike",
    "look-alike": "lookalike",
    "crm": "crm",
    "customer": "crm",
    "customer_match": "crm",
}


def normalize_categoricals(
    df: pd.DataFrame, applied_by: str | None = None
) -> tuple[pd.DataFrame, CleaningAction]:
    out = df.copy()
    before = out["audience_type"].astype(str).str.lower().str.strip().tolist()
    after = [_AUDIENCE_MAP.get(v, v) for v in before]
    out["audience_type"] = after
    affected = sum(1 for a, b in zip(before, after) if a != b)
    return out, CleaningAction(
        action_type="normalize_categoricals",
        rows_affected=affected,
        before_summary={
            "audience_type_unique_before": list(set(before)),
        },
        after_summary={
            "audience_type_unique_after": list(set(after)),
        },
        applied_by=applied_by,
    )


# --- Rule 3: normalize dates to ISO 8601 ------------------------------------


def normalize_dates_iso(
    df: pd.DataFrame, applied_by: str | None = None
) -> tuple[pd.DataFrame, CleaningAction]:
    out = df.copy()
    affected = 0
    for col in ("start_date", "end_date"):
        if col not in out.columns:
            continue
        before = out[col].astype(str).tolist()
        coerced = pd.to_datetime(out[col], errors="coerce")
        out[col] = coerced
        after = [d.isoformat() if pd.notnull(d) else None for d in coerced]
        affected += sum(
            1 for a, b in zip(before, after) if b and a != b
        )
    return out, CleaningAction(
        action_type="normalize_dates_iso",
        rows_affected=affected,
        applied_by=applied_by,
    )


# --- Rule 4: currency normalization to USD ----------------------------------


def normalize_currency_to_usd(
    df: pd.DataFrame,
    fx_rates: dict[str, float] | None = None,
    applied_by: str | None = None,
) -> tuple[pd.DataFrame, CleaningAction]:
    """Convert all spend to USD using configured FX rates (preserves original).

    If `currency_code` column is absent or all rows are USD, this is a no-op.
    fx_rates maps non-USD currency_code -> USD-per-unit.
    """
    out = df.copy()
    if "currency_code" not in out.columns:
        return out, CleaningAction(
            "normalize_currency_to_usd", 0,
            notes="No currency_code column; assumed USD.",
            applied_by=applied_by,
        )
    fx = fx_rates or {}
    out["cost_original"] = out["cost"]
    affected = 0
    for code, rate in fx.items():
        mask = out["currency_code"].astype(str).str.upper() == code.upper()
        if mask.any():
            out.loc[mask, "cost"] = out.loc[mask, "cost"] * rate
            affected += int(mask.sum())
    return out, CleaningAction(
        action_type="normalize_currency_to_usd",
        rows_affected=affected,
        before_summary={"unique_currencies": sorted(set(df["currency_code"].astype(str)))},
        applied_by=applied_by,
    )


# --- Rule 5: missing value policy -------------------------------------------


_CRITICAL_FIELDS = (
    "campaign_id", "advertiser_id", "is_rct", "vertical", "funnel_stage",
    "objective", "audience_type", "conversion_optimization", "custom_audience",
    "advertiser_platform_experience_months", "start_date", "end_date",
    "cost", "impressions", "clicks", "conversions", "lcc_7d",
)
_RECOMMENDED_FIELDS = (
    "lcc_1d", "lcc_28d", "view_through_conversions", "avg_dwell_time",
)


def handle_missing_values(
    df: pd.DataFrame, applied_by: str | None = None
) -> tuple[pd.DataFrame, CleaningAction]:
    """Reject critical-field nulls; impute recommended-field nulls (median)."""
    out = df.copy()
    rejected_rows: set[int] = set()
    for col in _CRITICAL_FIELDS:
        if col in out.columns:
            mask = out[col].isnull()
            rejected_rows.update(out[mask].index.tolist())
    out = out.drop(index=list(rejected_rows))

    imputed = 0
    for col in _RECOMMENDED_FIELDS:
        if col not in out.columns:
            continue
        mask = out[col].isnull()
        n = int(mask.sum())
        if n == 0:
            continue
        median_val = out[col].median(skipna=True)
        if pd.isnull(median_val):
            continue
        out.loc[mask, col] = median_val
        imputed += n

    return out, CleaningAction(
        action_type="handle_missing_values",
        rows_affected=len(rejected_rows) + imputed,
        before_summary={
            "rows_rejected_critical_null": len(rejected_rows),
            "rows_imputed_recommended": imputed,
        },
        applied_by=applied_by,
    )


# --- Rule 6: deduplication --------------------------------------------------


def dedupe(
    df: pd.DataFrame, applied_by: str | None = None
) -> tuple[pd.DataFrame, CleaningAction]:
    """Detect duplicates on campaign_id; keep latest by upload timestamp.

    If `uploaded_at` is absent, keep the last occurrence.
    """
    out = df.copy()
    n_before = len(out)
    sort_col = "uploaded_at" if "uploaded_at" in out.columns else None
    if sort_col:
        out = out.sort_values(sort_col)
    out = out.drop_duplicates(subset=["campaign_id"], keep="last")
    out = out.reset_index(drop=True)
    n_after = len(out)
    return out, CleaningAction(
        action_type="dedupe",
        rows_affected=n_before - n_after,
        applied_by=applied_by,
    )


# --- Rule 7: winsorization (off by default) --------------------------------


def winsorize(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    lower_pct: float = 1.0,
    upper_pct: float = 99.0,
    enabled: bool = False,
    applied_by: str | None = None,
) -> tuple[pd.DataFrame, CleaningAction]:
    """Optional 1st/99th-percentile winsorization."""
    out = df.copy()
    if not enabled:
        return out, CleaningAction(
            "winsorize", 0,
            notes="Winsorization disabled.", applied_by=applied_by,
        )
    cols = columns or [c for c in ("ctr", "cvr", "icpd", "cpic") if c in out.columns]
    affected = 0
    for col in cols:
        lo = np.percentile(out[col].dropna(), lower_pct)
        hi = np.percentile(out[col].dropna(), upper_pct)
        clipped_mask = (out[col] < lo) | (out[col] > hi)
        affected += int(clipped_mask.sum())
        out[col] = out[col].clip(lower=lo, upper=hi)
    return out, CleaningAction(
        action_type="winsorize",
        rows_affected=affected,
        notes=f"columns={cols} bounds=[{lower_pct}, {upper_pct}]",
        applied_by=applied_by,
    )


# --- Rule 8: hard-block invalid metric combinations -------------------------


def hard_block_invalid_metrics(
    df: pd.DataFrame, applied_by: str | None = None
) -> tuple[pd.DataFrame, CleaningAction]:
    """Drop rows that violate funnel monotonicity, cost > 0, or RCT control-exposure."""
    out = df.copy()
    n_before = len(out)
    bad = (
        (out["impressions"] < out["clicks"])
        | (out["clicks"] < out["conversions"])
        | (out["cost"] <= 0)
    )
    if "control_exposed_users" in out.columns:
        rct_mask = out["is_rct"] == 1
        bad |= rct_mask & (out["control_exposed_users"].fillna(0) > 0)
    out = out[~bad].reset_index(drop=True)
    n_after = len(out)
    return out, CleaningAction(
        action_type="hard_block_invalid_metrics",
        rows_affected=n_before - n_after,
        applied_by=applied_by,
    )


# --- Pipeline orchestration -------------------------------------------------


def run_pipeline(
    df: pd.DataFrame,
    *,
    applied_by: str | None = None,
    fx_rates: dict[str, float] | None = None,
    enable_winsorize: bool = False,
) -> tuple[pd.DataFrame, list[CleaningAction]]:
    """Run all eight cleaning rules in sequence; return (cleaned_df, audit log)."""
    actions: list[CleaningAction] = []
    out, a = standardize_campaign_ids(df, applied_by);     actions.append(a)
    out, a = normalize_categoricals(out, applied_by);      actions.append(a)
    out, a = normalize_dates_iso(out, applied_by);         actions.append(a)
    out, a = normalize_currency_to_usd(out, fx_rates, applied_by); actions.append(a)
    out, a = hard_block_invalid_metrics(out, applied_by);  actions.append(a)
    out, a = handle_missing_values(out, applied_by);       actions.append(a)
    out, a = dedupe(out, applied_by);                      actions.append(a)
    out, a = winsorize(out, enabled=enable_winsorize, applied_by=applied_by); actions.append(a)
    return out, actions
