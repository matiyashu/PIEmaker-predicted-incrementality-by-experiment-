"""
Schema mapping service (Prompt 1.1).

Auto-suggests mappings between user-uploaded column names and the standard
PIE upload schema (Appendix B). Returns confidence-scored candidates so the
analyst can override on the mapping UI.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

# Curated alias dictionary. Lowercased; punctuation/underscores stripped before lookup.
_ALIASES: dict[str, list[str]] = {
    "campaign_id": ["campaignid", "campaign", "cmpid", "campaign_key", "id"],
    "advertiser_id": ["advertiserid", "advertiser", "client_id", "account_id", "brand_id"],
    "campaign_name": ["campaignname", "name", "title"],
    "is_rct": ["israndomized", "rct", "is_experiment", "experiment"],
    "vertical": ["industry", "sector", "category"],
    "funnel_stage": ["funnel", "stage", "tofu_mofu_bofu"],
    "objective": ["goal", "primary_objective", "campaign_objective"],
    "audience_type": ["audience", "targeting", "audiencetype"],
    "conversion_optimization": ["convoptimization", "optimizationevent", "optimization"],
    "custom_audience": ["audience_breadth", "audience_size_class"],
    "advertiser_platform_experience_months": ["months_on_platform", "platform_tenure_months"],
    "start_date": ["startdate", "begin", "begin_date", "campaign_start"],
    "end_date": ["enddate", "finish", "stop_date", "campaign_end"],
    "cost": ["spend", "spent", "amount_spent", "media_cost"],
    "impressions": ["imp", "imps", "views"],
    "clicks": ["click", "linkclicks", "engagement_clicks"],
    "conversions": ["conv", "convs", "purchases", "actions"],
    "test_users": ["treatment_users", "exposed_group_size", "test_group_users"],
    "control_users": ["holdout_users", "control_group_users"],
    "exposed_test_users": ["treated_users", "actually_exposed_users"],
    "test_conversions": ["treatment_conversions", "test_outcomes"],
    "control_conversions": ["holdout_conversions", "control_outcomes"],
    "lcc_7d": ["last_click_7d", "lcc7d", "attributed_7d", "7d_click"],
    "lcc_1d": ["last_click_1d", "lcc1d", "attributed_1d", "1d_click"],
    "lcc_28d": ["last_click_28d", "lcc28d", "attributed_28d", "28d_click"],
    "lcc_1h": ["last_click_1h", "lcc1h"],
    "view_through_conversions": ["vtc", "view_through", "vt_conversions"],
    "avg_dwell_time": ["dwell", "average_dwell"],
    "conversion_value": ["revenue_per_conversion", "value_per_conversion", "aov"],
    "creative_format": ["format", "asset_type"],
    "placement": ["surface", "position"],
    "bid_strategy": ["bid_type", "buying_strategy"],
}


@dataclass
class MappingSuggestion:
    source_column: str
    target_field: str | None
    confidence: float
    reason: str


def _norm(s: str) -> str:
    return "".join(c for c in s.lower() if c.isalnum())


def _score(source_norm: str, target: str) -> tuple[float, str]:
    target_norm = _norm(target)
    if source_norm == target_norm:
        return 1.0, "exact match"
    aliases = [_norm(a) for a in _ALIASES.get(target, [])]
    if source_norm in aliases:
        return 0.95, "alias match"
    if source_norm in target_norm or target_norm in source_norm:
        return 0.65, "substring overlap"
    for alias in aliases:
        if source_norm in alias or alias in source_norm:
            return 0.55, "alias substring overlap"
    return 0.0, "no match"


def suggest_mappings(source_columns: list[str]) -> list[MappingSuggestion]:
    """Return one suggestion per source column (best target field, with score)."""
    suggestions: list[MappingSuggestion] = []
    used_targets: set[str] = set()
    for src in source_columns:
        src_norm = _norm(src)
        best: tuple[float, str | None, str] = (0.0, None, "no match")
        for target in list(_ALIASES.keys()):
            if target in used_targets:
                continue
            score, reason = _score(src_norm, target)
            if score > best[0]:
                best = (score, target, reason)
        if best[1] is not None and best[0] >= 0.5:
            used_targets.add(best[1])
        suggestions.append(
            MappingSuggestion(
                source_column=src,
                target_field=best[1] if best[0] >= 0.5 else None,
                confidence=best[0],
                reason=best[2],
            )
        )
    return suggestions


def apply_mapping(
    source_columns: list[str], mapping: dict[str, str]
) -> dict[str, str]:
    """Validate the user-confirmed mapping; raise ValueError on conflicts.

    Args:
        source_columns: columns from the uploaded file.
        mapping: user-confirmed {source_column: target_field}.

    Returns:
        A normalized mapping (subset of `mapping`) ready to apply with
        `df.rename(columns=mapping)`.
    """
    valid_targets = set(_ALIASES.keys())
    seen_targets: set[str] = set()
    out: dict[str, str] = {}
    for src, tgt in mapping.items():
        if src not in source_columns:
            raise ValueError(f"source column '{src}' not in uploaded file")
        if tgt not in valid_targets:
            raise ValueError(f"target field '{tgt}' is not a PIE schema column")
        if tgt in seen_targets:
            raise ValueError(f"target field '{tgt}' mapped more than once")
        seen_targets.add(tgt)
        out[src] = tgt
    return out


def to_dict_list(suggestions: list[MappingSuggestion]) -> list[dict]:
    return [asdict(s) for s in suggestions]
