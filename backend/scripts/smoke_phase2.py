"""End-to-end smoke for Phase 2: donor pool → labels → features → train → hold-out.

Runs against the JSON-file persistence shim. Resets state at start and end so
it does not pollute developer state across runs.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import pandas as pd

# Ensure repo root is on path when invoked as `python scripts/smoke_phase2.py`
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ml import model_registry  # noqa: E402
from ml.holdout_one_level import run_extrapolation_test  # noqa: E402
from services import donor_pool_service as dps  # noqa: E402
from services.feature_engineering_service import build_features  # noqa: E402
from services.label_generation_service import generate_labels  # noqa: E402
from services.model_training_service import train_pie_model  # noqa: E402
from services.persistence import read_table, reset  # noqa: E402

N_RCTS = 250  # research_mode band (≥200, <400)
VERTICALS = ["ecommerce", "travel", "finance", "media"]
AUDIENCES = ["retargeting", "prospecting", "lookalike"]
FUNNELS = ["upper", "mid", "lower"]


def _synthetic_rct(i: int, rng: random.Random) -> dict:
    test_users = rng.randint(800_000, 3_000_000)
    control_users = int(test_users * rng.uniform(0.85, 1.15))
    exposure = rng.uniform(0.6, 0.95)
    exposed = int(test_users * exposure)
    test_y = rng.uniform(0.018, 0.030)
    control_y = test_y * rng.uniform(0.78, 0.92)
    test_conv = int(test_users * test_y)
    control_conv = int(control_users * control_y)
    cost = rng.uniform(15_000, 250_000)
    duration = rng.choice([14, 21, 28, 35])
    vertical = rng.choice(VERTICALS)
    audience = rng.choice(AUDIENCES)
    funnel = rng.choice(FUNNELS)

    return {
        "campaign_id": f"CMP-{i:04d}",
        "advertiser_id": f"ADV-{i:04d}",
        "vertical": vertical,
        "audience_type": audience,
        "funnel_stage": funnel,
        "objective": "conversions",
        "conversion_optimization": rng.choice(["yes", "no"]),
        "custom_audience": rng.choice(["yes", "no"]),
        "advertiser_platform_experience_months": rng.randint(3, 60),
        "creative_format": rng.choice(["video", "image", "carousel"]),
        "placement": rng.choice(["feed", "story", "reels"]),
        "bid_strategy": rng.choice(["lowest_cost", "cost_cap", "bid_cap"]),
        "market": rng.choice(["US", "UK", "ID", "DE"]),
        "spend_tier": rng.choice(["low", "medium", "high"]),
        "platform": rng.choice(["meta", "tiktok", "google"]),
        "start_date": "2026-01-15",
        "end_date": f"2026-{rng.randint(1, 5):02d}-28",
        "test_users": test_users,
        "control_users": control_users,
        "exposed_test_users": exposed,
        "test_conversions": test_conv,
        "control_conversions": control_conv,
        "cost": cost,
        "duration_days": duration,
        "clicks": int(test_users * rng.uniform(0.01, 0.04)),
        "impressions": int(test_users * rng.uniform(2.0, 4.0)),
        "conversions": test_conv,
        "lcc_1h": int(test_conv * rng.uniform(0.02, 0.08)),
        "lcc_1d": int(test_conv * rng.uniform(0.05, 0.15)),
        "lcc_7d": int(test_conv * rng.uniform(0.10, 0.30)),
        "lcc_28d": int(test_conv * rng.uniform(0.20, 0.45)),
        "view_through_conversions": int(test_conv * rng.uniform(0.02, 0.10)),
        "avg_dwell_time": rng.uniform(4.0, 18.0),
    }


def main() -> int:
    print("=== Phase 2 Smoke ===")
    reset()
    for p in model_registry._REGISTRY_DIR.glob("*.pkl"):
        p.unlink()

    rng = random.Random(2026)
    rcts = [_synthetic_rct(i, rng) for i in range(N_RCTS)]
    print(f"  generated {len(rcts)} synthetic RCTs")

    # 1) Promote all RCTs to donor pool
    for r in rcts:
        dps.promote_rct(r["campaign_id"], r)
    pool_status = dps.get_pool_size_status()
    assert pool_status.band == "research_mode", pool_status
    print(f"  donor pool: {pool_status.n_admitted} admitted, band={pool_status.band}")

    # 2) Generate labels (ATT/IC/ICPD)
    labels_df = generate_labels(pd.DataFrame(rcts), has_user_level_data=False)
    assert len(labels_df) == N_RCTS
    print(f"  labels generated, mean ICPD={labels_df['icpd'].mean():.4f}")

    # 3) Build features
    feats_df = build_features(pd.DataFrame(rcts), mode="training")
    assert len(feats_df) == N_RCTS
    print(f"  features built, x_pre keys={len(feats_df.iloc[0]['x_pre'])}, "
          f"x_post keys={len(feats_df.iloc[0]['x_post'])}")

    # 4) Train (research-mode model)
    result = train_pie_model(name="smoke_pie_rf", n_bootstrap=50)
    print(
        f"  trained model {result['model']['id']}: "
        f"R²w={result['weighted_r_squared']:.3f}, "
        f"ceiling={result['r_squared_ceiling']:.3f}, "
        f"n={result['n_observations']}, "
        f"specs={[r['spec'] for r in result['ablation']]}"
    )
    assert result["model"]["status"] == "research"
    assert len(result["ablation"]) == 5

    # 5) Registry round-trip
    loaded = model_registry.load_model(result["model"]["id"])
    metrics = model_registry.metrics_for(result["model"]["id"])
    print(f"  registry: loaded {type(loaded).__name__}, {len(metrics)} metrics recorded")
    assert metrics, "expected metrics persisted"

    # 6) Hold-out-one-level on 'vertical'
    feat_rows = read_table("feature_store")
    label_rows = read_table("rct_labels")
    extrap = run_extrapolation_test(feat_rows, label_rows, "vertical", n_iterations=5)
    print(f"  hold-out (vertical): {len(extrap)} levels")
    for row in extrap:
        print(
            f"    {row['level']}: within={row['within_r2_median']:.2f}, "
            f"extrap={row['extrapolation_r2_median']:.2f}, "
            f"penalty={row['penalty_pp']:.1f}pp, risk={row['risk']}"
        )

    print("=== Phase 2 Smoke OK ===")
    reset()
    for p in model_registry._REGISTRY_DIR.glob("*.pkl"):
        p.unlink()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
