"""Generate a demo CSV for the dashboard.

Produces a synthetic dataset with:
  - 250 RCT rows (is_rct=1) — used to seed donor pool, generate labels,
    build training features, train a model
  - 50 non-RCT rows (is_rct=0) — campaigns waiting for an ICPD forecast,
    used by Portfolio / Decisions / Drift / Simulator

The schema covers every X_pre + X_post field the validation/feature
services expect, plus the date and user-count columns the RCT label
generator needs.

Usage:
    python scripts/generate_demo_csv.py [path]
Default output path: PIEMaker/demo/piemaker_demo.csv
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import pandas as pd

VERTICALS = ["ecommerce", "travel", "finance", "media"]
AUDIENCES = ["retargeting", "prospecting", "lookalike"]
FUNNELS = ["upper", "mid", "lower"]
CREATIVES = ["video", "image", "carousel"]
PLACEMENTS = ["feed", "story", "reels"]
BIDS = ["lowest_cost", "cost_cap", "bid_cap"]
MARKETS = ["US", "UK", "ID", "DE"]
SPEND = ["low", "medium", "high"]
PLATFORMS = ["meta", "tiktok", "google"]
# V.4: paper-aligned advertiser-size buckets and campaign_year range.
ADVERTISER_SIZES = ["smb", "mid_market", "enterprise"]
CAMPAIGN_YEARS = [2024, 2025, 2026]


def _row(i: int, *, rng: random.Random, is_rct: bool) -> dict:
    test_users = rng.randint(800_000, 3_000_000)
    control_users = (
        int(test_users * rng.uniform(0.85, 1.15)) if is_rct else 0
    )
    exposure = rng.uniform(0.60, 0.95)
    exposed = int(test_users * exposure)
    test_y = rng.uniform(0.018, 0.030)
    control_y = test_y * rng.uniform(0.78, 0.92)
    test_conv = int(test_users * test_y)
    control_conv = int(control_users * control_y) if is_rct else 0
    cost = rng.uniform(15_000, 250_000)
    duration = rng.choice([14, 21, 28, 35])
    # V.4: campaigns now span 3 calendar years so the hold-out-one-level
    # test on campaign_year (paper Table 1: 21pp drift penalty) can fire.
    year = rng.choice(CAMPAIGN_YEARS)
    start_month = rng.randint(1, 12) if year < 2026 else rng.randint(1, 5)
    start_date = f"{year}-{start_month:02d}-15"
    end_date = pd.Timestamp(start_date) + pd.Timedelta(days=duration)
    end_str = end_date.strftime("%Y-%m-%d")
    cid_prefix = "RCT" if is_rct else "CMP"

    return {
        "campaign_id": f"{cid_prefix}-{i:04d}",
        "advertiser_id": f"ADV-{i // 5:04d}",
        "is_rct": 1 if is_rct else 0,
        "objective": "conversions",
        "vertical": rng.choice(VERTICALS),
        "audience_type": rng.choice(AUDIENCES),
        "funnel_stage": rng.choice(FUNNELS),
        "conversion_optimization": rng.choice(["yes", "no"]),
        "custom_audience": rng.choice(["yes", "no"]),
        "advertiser_platform_experience_months": rng.randint(3, 60),
        # V.4 NEW fields
        "advertiser_size": rng.choice(ADVERTISER_SIZES),
        "campaign_year": year,
        "creative_format": rng.choice(CREATIVES),
        "placement": rng.choice(PLACEMENTS),
        "bid_strategy": rng.choice(BIDS),
        "market": rng.choice(MARKETS),
        "spend_tier": rng.choice(SPEND),
        "platform": rng.choice(PLATFORMS),
        "start_date": start_date,
        "end_date": end_str,
        "duration_days": duration,
        "test_users": test_users,
        "control_users": control_users,
        "exposed_test_users": exposed,
        "test_conversions": test_conv,
        "control_conversions": control_conv,
        "cost": round(cost, 2),
        "clicks": int(test_users * rng.uniform(0.01, 0.04)),
        "impressions": int(test_users * rng.uniform(2.0, 4.0)),
        "conversions": test_conv,
        "lcc_1h": int(test_conv * rng.uniform(0.02, 0.08)),
        "lcc_1d": int(test_conv * rng.uniform(0.05, 0.15)),
        "lcc_7d": int(test_conv * rng.uniform(0.10, 0.30)),
        "lcc_28d": int(test_conv * rng.uniform(0.20, 0.45)),
        "view_through_conversions": int(test_conv * rng.uniform(0.02, 0.10)),
        "avg_dwell_time": round(rng.uniform(4.0, 18.0), 2),
    }


def main(out_path: str | None = None) -> int:
    rng = random.Random(2026)
    rcts = [_row(i, rng=rng, is_rct=True) for i in range(400)]
    candidates = [_row(i, rng=rng, is_rct=False) for i in range(50)]

    df = pd.DataFrame(rcts + candidates)

    target = Path(
        out_path
        or Path(__file__).resolve().parents[2] / "demo" / "piemaker_demo.csv"
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(target, index=False)

    print(f"Wrote {len(df)} rows to {target}")
    print(f"  RCTs (is_rct=1):    {sum(df['is_rct'] == 1)}")
    print(f"  Non-RCTs (is_rct=0): {sum(df['is_rct'] == 0)}")
    print(f"  Columns: {len(df.columns)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else None))
