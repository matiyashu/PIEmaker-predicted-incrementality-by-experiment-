"""
V.4 Wave 5 (Phase 8) — edge-case fixture builders.

Six synthetic donor pools, each deliberately tuned to trip a specific V.4
diagnostic. Tests in ``tests/test_v4_fixtures.py`` load these and assert
the expected band fires.

Each builder returns ``(feature_rows, label_rows, cost_weights)`` in the
shape the existing services expect (matching test_oof_evaluation and
test_wave2_diagnostics conventions).

Fixtures:
  * perfect_compliance        — D_bar = 1.0 everywhere
  * one_sided_noncompliance   — D_bar < 1 with non-trivial spread
  * sample_split              — has_user_level_data path active
  * high_lcc_bias             — Raw LCC-7D consistently overstates ICPD
  * cold_start_advertisers    — 50% of rows are single-campaign advertisers
  * severe_extrapolation      — one vertical level entirely held out from training
"""

from __future__ import annotations

import random
from typing import Literal

FixtureName = Literal[
    "perfect_compliance",
    "one_sided_noncompliance",
    "sample_split",
    "high_lcc_bias",
    "cold_start_advertisers",
    "severe_extrapolation",
]


def _base_x_pre(rng: random.Random, *, vertical: str | None = None) -> dict:
    return {
        "objective": "conversions",
        "vertical": vertical if vertical is not None else rng.choice(
            ["ecommerce", "travel", "finance"]
        ),
        "audience_type": rng.choice(["retargeting", "prospecting", "lookalike"]),
        "funnel_stage": rng.choice(["upper", "mid", "lower"]),
        "conversion_optimization": rng.choice(["yes", "no"]),
        "custom_audience": rng.choice(["yes", "no"]),
        "advertiser_platform_experience_months": rng.randint(6, 60),
        "advertiser_id": "ADV-000",  # caller overrides
        "advertiser_size": rng.choice(["smb", "mid_market", "enterprise"]),
        "campaign_year": rng.choice([2024, 2025, 2026]),
        "creative_format": "video",
        "placement": "feed",
        "bid_strategy": "lowest_cost",
        "market": "US",
        "spend_tier": rng.choice(["low", "medium", "high"]),
        "platform": "meta",
        "month": rng.randint(1, 12),
        "quarter": rng.randint(1, 4),
        "campaign_duration_days": rng.choice([14, 21, 28, 35]),
    }


def _base_x_post(cpd: float, lcc7: float, exposure: float = 0.8) -> dict:
    return {
        "exposure_rate": exposure,
        "ctr": 0.02,
        "clicks_per_dollar": 2.0,
        "conversions_per_dollar": cpd,
        "lcc_1h_per_dollar": 0.01,
        "lcc_1d_per_dollar": 0.02,
        "lcc_7d_per_dollar": lcc7,
        "lcc_28d_per_dollar": 0.08,
        "view_through_per_dollar": 0.01,
        "avg_dwell_time": 10.0,
    }


def _make_row(
    i: int,
    rng: random.Random,
    *,
    icpd: float,
    cost: float = 50_000.0,
    advertiser_id: str | None = None,
    vertical: str | None = None,
    exposure: float = 0.8,
) -> tuple[dict, dict]:
    cid = f"CMP-{i:04d}"
    cpd = rng.uniform(0.5, 5.0)
    lcc7 = rng.uniform(0.05, 0.5)
    x_pre = _base_x_pre(rng, vertical=vertical)
    if advertiser_id is not None:
        x_pre["advertiser_id"] = advertiser_id
    feat = {
        "campaign_id": cid,
        "cost": cost,
        "x_pre": x_pre,
        "x_post": _base_x_post(cpd, lcc7, exposure=exposure),
    }
    label = {"campaign_id": cid, "icpd": icpd, "cost": cost}
    return feat, label


# --- Fixtures ----------------------------------------------------------------


def perfect_compliance(
    n: int = 30, seed: int = 1
) -> tuple[list[dict], list[dict], list[float]]:
    """D_bar = 1.0 for every RCT — every test user actually saw the ad."""
    rng = random.Random(seed)
    feats: list[dict] = []
    labels: list[dict] = []
    weights: list[float] = []
    for i in range(n):
        cpd = rng.uniform(0.5, 5.0)
        lcc7 = rng.uniform(0.05, 0.5)
        icpd_val = 0.3 * cpd + 1.2 * lcc7 + rng.gauss(0, 0.02)
        cost = rng.uniform(20_000, 200_000)
        feat, label = _make_row(
            i, rng, icpd=icpd_val, cost=cost, exposure=1.0
        )
        feats.append(feat)
        labels.append(label)
        weights.append(cost)
    return feats, labels, weights


def one_sided_noncompliance(
    n: int = 30, seed: int = 2
) -> tuple[list[dict], list[dict], list[float]]:
    """D_bar in [0.40, 0.85] — many test users were not exposed."""
    rng = random.Random(seed)
    feats: list[dict] = []
    labels: list[dict] = []
    weights: list[float] = []
    for i in range(n):
        cpd = rng.uniform(0.5, 5.0)
        lcc7 = rng.uniform(0.05, 0.5)
        icpd_val = 0.3 * cpd + 1.2 * lcc7 + rng.gauss(0, 0.02)
        cost = rng.uniform(20_000, 200_000)
        feat, label = _make_row(
            i, rng, icpd=icpd_val, cost=cost, exposure=rng.uniform(0.40, 0.85)
        )
        feats.append(feat)
        labels.append(label)
        weights.append(cost)
    return feats, labels, weights


def sample_split(
    n: int = 30, seed: int = 3
) -> tuple[list[dict], list[dict], list[float]]:
    """label rows carry has_user_level_data=True markers + Sample-1/Sample-2
    populated so the label-noise R² ceiling can be computed."""
    rng = random.Random(seed)
    feats: list[dict] = []
    labels: list[dict] = []
    weights: list[float] = []
    for i in range(n):
        cpd = rng.uniform(0.5, 5.0)
        lcc7 = rng.uniform(0.05, 0.5)
        icpd_val = 0.3 * cpd + 1.2 * lcc7 + rng.gauss(0, 0.02)
        cost = rng.uniform(20_000, 200_000)
        feat, label = _make_row(
            i, rng, icpd=icpd_val, cost=cost, exposure=0.85
        )
        # Inject the V.4 per-arm metadata the label-noise R² ceiling reads.
        s1_test_users = 1_000_000.0
        s1_control_users = 1_000_000.0
        s1_test_conv = 12_000.0
        s1_control_conv = 9_500.0
        label.update(
            {
                "sample_split_active": True,
                "sample_split_seed": 42 + i,
                "sample_1_test_users": s1_test_users,
                "sample_1_control_users": s1_control_users,
                "sample_1_test_conversions": s1_test_conv,
                "sample_1_control_conversions": s1_control_conv,
                "sample_1_exposed_test_users": 850_000.0,
                "sample_2_test_users": s1_test_users,
                "sample_2_control_users": s1_control_users,
                "sample_2_exposed_test_users": 850_000.0,
                "sample_2_test_conversions": s1_test_conv,
                "sample_2_control_conversions": s1_control_conv,
                "exposure_rate": 0.85,
            }
        )
        feats.append(feat)
        labels.append(label)
        weights.append(cost)
    return feats, labels, weights


def high_lcc_bias(
    n: int = 40, seed: int = 4
) -> tuple[list[dict], list[dict], list[float]]:
    """Raw LCC-7D massively overstates ICPD — calibration should flag
    bias_ratio > 1.5 on every segment."""
    rng = random.Random(seed)
    feats: list[dict] = []
    labels: list[dict] = []
    weights: list[float] = []
    for i in range(n):
        cpd = rng.uniform(0.5, 5.0)
        # LCC-7D-per-dollar is 3× the true ICPD signal — the bias.
        lcc7 = rng.uniform(0.3, 0.8)
        icpd_val = 0.05 + 0.02 * cpd + rng.gauss(0, 0.005)
        cost = rng.uniform(20_000, 200_000)
        feat, label = _make_row(
            i, rng, icpd=icpd_val, cost=cost
        )
        feat["x_post"]["lcc_7d_per_dollar"] = lcc7
        feats.append(feat)
        labels.append(label)
        weights.append(cost)
    return feats, labels, weights


def cold_start_advertisers(
    n_existing_adv: int = 8,
    campaigns_per_existing: int = 5,
    n_new_adv: int = 25,
    seed: int = 5,
) -> tuple[list[dict], list[dict], list[float]]:
    """Mix of existing advertisers (multiple campaigns) + new advertisers
    (one campaign each). Advertiser CV should show a non-trivial cohort gap."""
    rng = random.Random(seed)
    feats: list[dict] = []
    labels: list[dict] = []
    weights: list[float] = []
    i = 0
    for adv in range(n_existing_adv):
        for _ in range(campaigns_per_existing):
            cpd = rng.uniform(0.5, 5.0)
            lcc7 = rng.uniform(0.05, 0.5)
            icpd_val = 0.3 * cpd + 1.2 * lcc7 + rng.gauss(0, 0.02)
            cost = rng.uniform(20_000, 200_000)
            feat, label = _make_row(
                i, rng, icpd=icpd_val, cost=cost,
                advertiser_id=f"ADV-EXISTING-{adv:02d}",
            )
            feats.append(feat)
            labels.append(label)
            weights.append(cost)
            i += 1
    for adv in range(n_new_adv):
        cpd = rng.uniform(0.5, 5.0)
        lcc7 = rng.uniform(0.05, 0.5)
        # Cold-start advertisers have a different ICPD prior — that's
        # exactly what makes them risky for the model.
        icpd_val = 0.2 * cpd + 0.8 * lcc7 + rng.gauss(0, 0.05)
        cost = rng.uniform(20_000, 200_000)
        feat, label = _make_row(
            i, rng, icpd=icpd_val, cost=cost,
            advertiser_id=f"ADV-NEW-{adv:02d}",
        )
        feats.append(feat)
        labels.append(label)
        weights.append(cost)
        i += 1
    return feats, labels, weights


def severe_extrapolation(
    n_majority: int = 40, n_held_out: int = 10, seed: int = 6
) -> tuple[list[dict], list[dict], list[float]]:
    """A donor pool where one vertical (`media`) sits at a different ICPD
    regime — hold-out-one-level should flag severe penalty on it."""
    rng = random.Random(seed)
    feats: list[dict] = []
    labels: list[dict] = []
    weights: list[float] = []
    i = 0
    # Majority: ecommerce / travel / finance, normal ICPD signal
    for _ in range(n_majority):
        cpd = rng.uniform(0.5, 5.0)
        lcc7 = rng.uniform(0.05, 0.5)
        icpd_val = 0.3 * cpd + 1.2 * lcc7 + rng.gauss(0, 0.02)
        cost = rng.uniform(20_000, 200_000)
        feat, label = _make_row(
            i, rng, icpd=icpd_val, cost=cost,
            vertical=rng.choice(["ecommerce", "travel", "finance"]),
        )
        feats.append(feat)
        labels.append(label)
        weights.append(cost)
        i += 1
    # Held-out level: `media`, with a wildly different ICPD relationship
    for _ in range(n_held_out):
        cpd = rng.uniform(0.5, 5.0)
        lcc7 = rng.uniform(0.05, 0.5)
        # Negative coefficient + offset — model trained on the majority
        # won't be able to predict media correctly.
        icpd_val = 0.8 - 0.15 * cpd - 0.3 * lcc7 + rng.gauss(0, 0.05)
        cost = rng.uniform(20_000, 200_000)
        feat, label = _make_row(
            i, rng, icpd=icpd_val, cost=cost, vertical="media",
        )
        feats.append(feat)
        labels.append(label)
        weights.append(cost)
        i += 1
    return feats, labels, weights


def load(name: FixtureName) -> tuple[list[dict], list[dict], list[float]]:
    """Convenience dispatcher."""
    builders = {
        "perfect_compliance": perfect_compliance,
        "one_sided_noncompliance": one_sided_noncompliance,
        "sample_split": sample_split,
        "high_lcc_bias": high_lcc_bias,
        "cold_start_advertisers": cold_start_advertisers,
        "severe_extrapolation": severe_extrapolation,
    }
    return builders[name]()
