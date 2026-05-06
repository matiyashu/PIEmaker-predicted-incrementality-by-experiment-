"""
RCT Label Generator (Prompt 2.2).

For each admitted RCT, computes ATT (Eq. 14), IC (Eq. 22), ICPD (Eq. 23) using
the frozen formulas in pie_formulas/. Tags each row with its mc_defense_mode
and (when sample_split applies) records the deterministic seed.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from pie_formulas import att, exposure_rate, icpd, incremental_conversions
from services.mechanical_correlation_defense import decide
from services.persistence import upsert

LABELS_TABLE = "rct_labels"


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
        d = decide(
            campaign_id=cid,
            has_user_level_data=has_user_level_data,
            test_users=float(row["test_users"]),
        )
        d_tr = exposure_rate(
            float(row["exposed_test_users"]), float(row["test_users"])
        )
        att_val = att(
            test_conversions=float(row["test_conversions"]),
            test_users=float(row["test_users"]),
            control_conversions=float(row["control_conversions"]),
            control_users=float(row["control_users"]),
            exposure_rate_value=d_tr,
        )
        ic = incremental_conversions(att_val, d_tr, float(row["test_users"]))
        icpd_val = icpd(ic, float(row["cost"]))

        record = {
            "campaign_id": cid,
            "att": att_val,
            "incremental_conversions": ic,
            "icpd": icpd_val,
            "exposure_rate": d_tr,
            "mc_defense_mode": d.mc_defense_mode,
            "sample_split_seed": d.sample_split_seed,
            "admitted_to_donor_pool": False,  # promoted via Donor Pool service
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        upsert(LABELS_TABLE, record, key="campaign_id")
        out.append(record)
    return pd.DataFrame(out)
