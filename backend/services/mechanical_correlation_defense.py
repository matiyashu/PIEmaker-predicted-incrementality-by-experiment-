"""
Mechanical-correlation defense (PDF §4.4, Prompt 1.2).

Decides per RCT whether to:
  (a) sample-split:    user-level data available — gold standard
  (b) shared-sample:   ≥1M aggregated users — flagged compromise
  (c) block:           <1M aggregated users — research-mode only

For sample-split: a deterministic 50/50 random partition of test users keyed
on a hash of campaign_id, so the split is reproducible across runs.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np
import pandas as pd

DefenseMode = Literal["sample_split", "shared_sample_compromise", "blocked"]
MIN_USERS_FOR_AGGREGATED = 1_000_000


@dataclass
class DefenseDecision:
    campaign_id: str
    mc_defense_mode: DefenseMode
    sample_split_seed: int | None
    reason: str

    def to_dict(self) -> dict:
        return asdict(self)


def _seed_from_campaign(campaign_id: str) -> int:
    """Deterministic 64-bit seed from campaign_id."""
    h = hashlib.sha256(campaign_id.encode("utf-8")).digest()
    return int.from_bytes(h[:8], byteorder="big", signed=False)


def decide(
    campaign_id: str,
    has_user_level_data: bool,
    test_users: float,
) -> DefenseDecision:
    """Pick the defense mode for a single RCT."""
    if has_user_level_data:
        return DefenseDecision(
            campaign_id=campaign_id,
            mc_defense_mode="sample_split",
            sample_split_seed=_seed_from_campaign(campaign_id),
            reason=(
                "User-level data available; applying 50/50 sample-split "
                "(PDF §4.4, gold standard)."
            ),
        )
    if test_users >= MIN_USERS_FOR_AGGREGATED:
        return DefenseDecision(
            campaign_id=campaign_id,
            mc_defense_mode="shared_sample_compromise",
            sample_split_seed=None,
            reason=(
                f"Aggregated data with {int(test_users):,} test users (≥ "
                f"{MIN_USERS_FOR_AGGREGATED:,}); shared-sample compromise mode. "
                "Documented in Model Card."
            ),
        )
    return DefenseDecision(
        campaign_id=campaign_id,
        mc_defense_mode="blocked",
        sample_split_seed=None,
        reason=(
            f"Aggregated data with only {int(test_users):,} test users "
            f"(< {MIN_USERS_FOR_AGGREGATED:,}); mechanical correlation cannot "
            "be empirically tested at this scale. Research mode only."
        ),
    )


def split_users(
    n_users: int, campaign_id: str
) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic 50/50 partition of user indices for sample-splitting.

    Returns (sample_1_idx, sample_2_idx). Same campaign_id produces identical
    partitions on rerun.
    """
    if n_users < 2:
        raise ValueError("need at least 2 users to sample-split")
    rng = np.random.default_rng(_seed_from_campaign(campaign_id))
    perm = rng.permutation(n_users)
    half = n_users // 2
    return perm[:half], perm[half:]


def decide_for_dataframe(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Apply `decide()` to every RCT row; non-RCT rows return None.

    Required columns: campaign_id, is_rct, test_users, has_user_level_data
    (the last is optional; defaults to False if missing).
    """
    out = df.copy()
    has_user_level = (
        out["has_user_level_data"]
        if "has_user_level_data" in out.columns
        else pd.Series([False] * len(out), index=out.index)
    )
    decisions: list[dict | None] = []
    for idx, row in out.iterrows():
        if int(row["is_rct"]) != 1:
            decisions.append(None)
            continue
        d = decide(
            campaign_id=str(row["campaign_id"]),
            has_user_level_data=bool(has_user_level.iloc[idx] if hasattr(has_user_level, "iloc") else has_user_level[idx]),
            test_users=float(row["test_users"]),
        )
        decisions.append(d.to_dict())
    out["mc_defense"] = decisions
    return out
