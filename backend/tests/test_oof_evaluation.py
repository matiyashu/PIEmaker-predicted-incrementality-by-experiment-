"""V.4 Wave 1 — OOF evaluation, true cost weights, label-noise R² ceiling."""

from __future__ import annotations

import math
import random

import numpy as np
import pytest

from ml.train_random_forest import train
from pie_formulas import (
    att_variance,
    icpd_label_variance,
    r_squared_ceiling_from_label_noise,
    weighted_r_squared,
)


def _synth_rows(n: int = 30, seed: int = 7) -> tuple[list[dict], list[dict]]:
    rng = random.Random(seed)
    feats: list[dict] = []
    labels: list[dict] = []
    for i in range(n):
        cid = f"CMP-{i:03d}"
        cpd = rng.uniform(0.5, 5.0)
        lcc7 = rng.uniform(0.05, 0.5)
        icpd_val = 0.3 * cpd + 1.2 * lcc7 + rng.gauss(0, 0.02)
        feats.append(
            {
                "campaign_id": cid,
                "x_pre": {
                    "objective": "conversions",
                    "vertical": rng.choice(["ecommerce", "travel", "finance"]),
                    "audience_type": "retargeting",
                    "funnel_stage": "lower",
                    "conversion_optimization": "yes",
                    "custom_audience": "yes",
                    "advertiser_platform_experience_months": 12,
                    "advertiser_id": f"ADV-{i // 5:03d}",
                    "advertiser_size": rng.choice(["smb", "mid_market", "enterprise"]),
                    "campaign_year": rng.choice([2024, 2025, 2026]),
                    "creative_format": "video",
                    "placement": "feed",
                    "bid_strategy": "lowest_cost",
                    "market": "US",
                    "spend_tier": "high",
                    "platform": "meta",
                    "month": 3,
                    "quarter": 1,
                    "campaign_duration_days": 21,
                },
                "x_post": {
                    "exposure_rate": 0.8,
                    "ctr": 0.02,
                    "clicks_per_dollar": 2.0,
                    "conversions_per_dollar": cpd,
                    "lcc_1h_per_dollar": 0.01,
                    "lcc_1d_per_dollar": 0.02,
                    "lcc_7d_per_dollar": lcc7,
                    "lcc_28d_per_dollar": 0.08,
                    "view_through_per_dollar": 0.01,
                    "avg_dwell_time": 10.0,
                },
            }
        )
        labels.append({"campaign_id": cid, "icpd": icpd_val})
    return feats, labels


# --- train() now returns OOF ---------------------------------------------------


def test_train_returns_oof_predictions_by_default():
    feats, labels = _synth_rows(30)
    artifacts = train(feats, labels, n_splits=5)  # cap CV for speed
    assert artifacts.oof_predictions is not None
    assert artifacts.oof_campaign_ids is not None
    assert artifacts.oof_n_splits == 5
    assert len(artifacts.oof_predictions) == 30
    assert len(artifacts.oof_campaign_ids) == 30
    # Every row covered by exactly one held-out fold → no NaNs.
    assert not any(math.isnan(v) for v in artifacts.oof_predictions)


def test_train_oof_disabled_returns_none():
    feats, labels = _synth_rows(20)
    artifacts = train(feats, labels, n_splits=4, return_oof=False)
    assert artifacts.oof_predictions is None
    assert artifacts.oof_campaign_ids is None
    assert artifacts.oof_n_splits is None


def test_oof_predictions_are_out_of_sample_not_in_sample():
    """OOF R² should differ materially from in-sample R² on a single
    well-fit pool — that's the whole point of using OOF."""
    feats, labels = _synth_rows(40, seed=11)
    artifacts = train(feats, labels, n_splits=5)
    icpd_true = []
    for cid in artifacts.oof_campaign_ids or []:
        match = next(l for l in labels if l["campaign_id"] == cid)
        icpd_true.append(match["icpd"])
    w = [1.0] * len(icpd_true)
    oof_r2 = weighted_r_squared(icpd_true, artifacts.oof_predictions, w)

    # In-sample R² on the refit model: this is what V.3 reported.
    from ml.train_random_forest import _flatten_feature_rows
    flat = _flatten_feature_rows(feats).set_index("campaign_id")
    in_sample_preds = artifacts.estimator.predict(flat.loc[artifacts.oof_campaign_ids]).tolist()
    in_sample_r2 = weighted_r_squared(icpd_true, in_sample_preds, w)

    # The two must be finite floats. In-sample is typically much higher
    # than OOF on small synthetic pools (>0.05 gap is realistic).
    assert math.isfinite(oof_r2)
    assert math.isfinite(in_sample_r2)
    assert in_sample_r2 - oof_r2 > 0.05


# --- Label-noise R² ceiling (paper §5.2 fn. 21) -------------------------------


def test_att_variance_positive_for_typical_rct():
    var = att_variance(
        test_conversions=10_000,
        test_users=1_000_000,
        control_conversions=8_000,
        control_users=1_000_000,
        exposure_rate_value=0.8,
    )
    assert var > 0
    assert math.isfinite(var)


def test_att_variance_rejects_zero_arms():
    with pytest.raises(ValueError):
        att_variance(1, 0, 1, 1, 0.5)
    with pytest.raises(ValueError):
        att_variance(1, 1, 1, 1, 0.0)


def test_icpd_label_variance_scales_with_cost():
    """ICPD variance falls quadratically with cost — bigger spend, less
    relative noise."""
    args = dict(
        test_conversions=10_000,
        test_users=1_000_000,
        control_conversions=8_000,
        control_users=1_000_000,
        exposure_rate_value=0.8,
    )
    small = icpd_label_variance(**args, cost=10_000)
    big = icpd_label_variance(**args, cost=100_000)
    assert big < small
    # The 10× cost ratio should reduce ICPD variance by ~100× (cost²).
    assert small / big == pytest.approx(100.0, rel=1e-6)


def test_r2_ceiling_from_label_noise_clamps_to_unit_interval():
    # All labels equal → no signal → ceiling is undefined; return 1.0.
    assert r_squared_ceiling_from_label_noise([0.1], [0.5]) == 1.0

    # Label noise tiny vs total variance → ceiling close to 1.0.
    labels = list(np.linspace(0.0, 1.0, 20))
    tiny_var = [1e-6] * 20
    high_ceiling = r_squared_ceiling_from_label_noise(tiny_var, labels)
    assert high_ceiling > 0.99

    # Label noise comparable to total → ceiling close to 0.
    big_var = [0.5] * 20  # variance ~ total variance of labels
    low_ceiling = r_squared_ceiling_from_label_noise(big_var, labels)
    assert 0.0 <= low_ceiling <= 1.0


def test_r2_ceiling_label_noise_uses_weights():
    """Heavily weighting low-noise rows should raise the ceiling."""
    labels = [0.0, 0.5, 1.0]
    variances = [1.0, 0.001, 0.001]  # row 0 is noisy
    unw = r_squared_ceiling_from_label_noise(variances, labels)
    weighted_away_from_noisy = r_squared_ceiling_from_label_noise(
        variances, labels, weights=[0.01, 1.0, 1.0]
    )
    assert weighted_away_from_noisy > unw


def test_r2_ceiling_label_noise_rejects_misshaped_inputs():
    with pytest.raises(ValueError):
        r_squared_ceiling_from_label_noise([], [])
    with pytest.raises(ValueError):
        r_squared_ceiling_from_label_noise([0.1, 0.2], [0.5])
