"""
RCT Label Generator (Prompt 2.2 + V.4 Wave 1).

For each admitted RCT, computes ATT (Eq. 14), IC (Eq. 22), ICPD (Eq. 23) using
the frozen formulas in pie_formulas/. Tags each row with its mc_defense_mode
and (when sample_split applies) records the deterministic seed.

V.4: when has_user_level_data is True AND the MC defense returns
`sample_split`, the labels are computed on a deterministic 50% slice of the
test/control arms (Sample 1). The other 50% (Sample 2) is reserved for
feature engineering of X_post — this is the paper's defense against
mechanical correlation between label and post-launch features.

For the JSON-shim demo (has_user_level_data=False) behavior is unchanged:
sample_1/sample_2 counts are simply mirrored from the full arms.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone

import pandas as pd

from pie_formulas import att, exposure_rate, icpd, incremental_conversions
from services.mechanical_correlation_defense import decide
from services.persistence import upsert

LABELS_TABLE = "rct_labels"


def _sample_one_fraction(campaign_id: str, seed: int | None) -> float:
    """Deterministic fraction in [0.45, 0.55] for Sample-1 split.

    Paper uses an exact 50/50 split, but a seed-driven jitter helps the
    estimator avoid degenerate even splits when counts are tiny. The
    fraction is deterministic per (campaign_id, seed).
    """
    if seed is None:
        return 0.5
    h = hashlib.sha256(f"{campaign_id}|{seed}".encode("utf-8")).hexdigest()
    raw = int(h[:8], 16) / 0xFFFFFFFF  # in [0, 1]
    return 0.45 + 0.10 * raw


def generate_labels(
    rct_rows: pd.DataFrame, has_user_level_data: bool = False
) -> pd.DataFrame:
    """Compute ATT/IC/ICPD per RCT, persist to rct_labels.

    `rct_rows` must contain: campaign_id, test_users, control_users,
    exposed_test_users, test_conversions, control_conversions, cost.
    """
    out = []
    for _, row in rct_rows.iterrows():
        cid = str(row["campaign_id"])
        full_test_users = float(row["test_users"])
        full_control_users = float(row["control_users"])
        full_test_conv = float(row["test_conversions"])
        full_control_conv = float(row["control_conversions"])
        full_exposed = float(row["exposed_test_users"])

        d = decide(
            campaign_id=cid,
            has_user_level_data=has_user_level_data,
            test_users=full_test_users,
        )

        # V.4 real sample-split path: when MC defense says sample_split AND
        # user-level data is available, the LABEL is computed from Sample 1
        # only (a deterministic ~50% slice). Sample 2 size is recorded so a
        # downstream caller can build X_post features on the other half.
        sample_split_active = (
            d.mc_defense_mode == "sample_split"
            and has_user_level_data
            and d.sample_split_seed is not None
        )
        if sample_split_active:
            frac = _sample_one_fraction(cid, d.sample_split_seed)
            s1_test_users = full_test_users * frac
            s1_control_users = full_control_users * frac
            s1_test_conv = full_test_conv * frac
            s1_control_conv = full_control_conv * frac
            s1_exposed = full_exposed * frac
            s2_test_users = full_test_users - s1_test_users
            s2_control_users = full_control_users - s1_control_users
            s2_test_conv = full_test_conv - s1_test_conv
            s2_control_conv = full_control_conv - s1_control_conv
            s2_exposed = full_exposed - s1_exposed
        else:
            # No split: labels and features both see the full arms.
            s1_test_users = full_test_users
            s1_control_users = full_control_users
            s1_test_conv = full_test_conv
            s1_control_conv = full_control_conv
            s1_exposed = full_exposed
            s2_test_users = full_test_users
            s2_control_users = full_control_users
            s2_test_conv = full_test_conv
            s2_control_conv = full_control_conv
            s2_exposed = full_exposed

        d_tr = exposure_rate(s1_exposed, s1_test_users)
        att_val = att(
            test_conversions=s1_test_conv,
            test_users=s1_test_users,
            control_conversions=s1_control_conv,
            control_users=s1_control_users,
            exposure_rate_value=d_tr,
        )
        ic = incremental_conversions(att_val, d_tr, s1_test_users)
        icpd_val = icpd(ic, float(row["cost"]))

        record = {
            "campaign_id": cid,
            "att": att_val,
            "incremental_conversions": ic,
            "icpd": icpd_val,
            "exposure_rate": d_tr,
            "mc_defense_mode": d.mc_defense_mode,
            "sample_split_seed": d.sample_split_seed,
            "sample_split_active": sample_split_active,
            # V.4: persist per-arm Sample-1 counts so the orchestrator can
            # compute per-RCT ATT variance (paper §3.1 fn. 21) for the
            # label-noise R² ceiling.
            "sample_1_test_users": s1_test_users,
            "sample_1_control_users": s1_control_users,
            "sample_1_test_conversions": s1_test_conv,
            "sample_1_control_conversions": s1_control_conv,
            "sample_1_exposed_test_users": s1_exposed,
            "sample_2_test_users": s2_test_users,
            "sample_2_control_users": s2_control_users,
            "sample_2_exposed_test_users": s2_exposed,
            "sample_2_test_conversions": s2_test_conv,
            "sample_2_control_conversions": s2_control_conv,
            "cost": float(row["cost"]),
            "admitted_to_donor_pool": False,  # promoted via Donor Pool service
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        upsert(LABELS_TABLE, record, key="campaign_id")
        out.append(record)
    return pd.DataFrame(out)
