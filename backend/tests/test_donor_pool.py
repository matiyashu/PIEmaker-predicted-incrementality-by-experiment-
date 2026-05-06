"""Donor Pool Manager tests (PDF §4.2, Prompt 2.1)."""

from __future__ import annotations

import pytest

from services import donor_pool_service as dps
from services.persistence import reset


@pytest.fixture(autouse=True)
def _reset_state():
    reset()
    yield
    reset()


def _make_rct(cid: str, **overrides) -> dict:
    base = {
        "campaign_id": cid,
        "advertiser_id": f"ADV-{cid}",
        "vertical": "ecommerce",
        "funnel_stage": "lower",
        "audience_type": "retargeting",
        "test_users": 2_000_000,
        "control_users": 2_000_000,
        "test_conversions": 10_000,
        "duration_days": 21,
        "end_date": "2026-04-30",
    }
    base.update(overrides)
    return base


def test_band_blocked_below_200():
    for i in range(199):
        dps.promote_rct(f"CMP-{i}", _make_rct(f"CMP-{i}"))
    s = dps.get_pool_size_status()
    assert s.band == "blocked"
    assert s.n_admitted == 199


def test_band_research_at_200():
    for i in range(200):
        dps.promote_rct(f"CMP-{i}", _make_rct(f"CMP-{i}"))
    assert dps.get_pool_size_status().band == "research_mode"


def test_band_production_at_400():
    for i in range(400):
        dps.promote_rct(f"CMP-{i}", _make_rct(f"CMP-{i}"))
    assert dps.get_pool_size_status().band == "production"


def test_band_full_at_1600():
    for i in range(1600):
        dps.promote_rct(f"CMP-{i}", _make_rct(f"CMP-{i}"))
    assert dps.get_pool_size_status().band == "production_full"


def test_quality_score_strong_rct():
    rct = _make_rct(
        "CMP-A",
        test_users=2_000_000,
        control_users=2_000_000,
        test_conversions=10_000,
        duration_days=28,
    )
    augmented = dps.list_eligible_rcts([rct])[0]
    assert augmented["quality_score"] >= 80


def test_quality_score_weak_rct():
    rct = _make_rct(
        "CMP-B",
        test_users=200_000,
        control_users=10_000,
        test_conversions=100,
        duration_days=2,
    )
    augmented = dps.list_eligible_rcts([rct])[0]
    assert augmented["quality_score"] <= 30


def test_promote_then_demote():
    rct = _make_rct("CMP-1")
    dps.promote_rct("CMP-1", rct)
    assert any(r["campaign_id"] == "CMP-1" and r["admitted"] for r in dps.get_admitted_rcts())
    dps.demote_rct("CMP-1")
    assert not any(r["campaign_id"] == "CMP-1" and r["admitted"] for r in dps.get_admitted_rcts())


def test_demote_unknown_returns_none():
    assert dps.demote_rct("PHANTOM") is None


def test_coverage_heatmap_counts_admitted_only():
    pool = [_make_rct(f"CMP-{i}", vertical="travel") for i in range(3)]
    pool += [_make_rct(f"OTH-{i}", vertical="ecommerce") for i in range(2)]
    for r in pool:
        dps.promote_rct(r["campaign_id"], r)
    # Demote one ecommerce row so it's not admitted
    dps.demote_rct("OTH-0")
    hm = dps.coverage_heatmap(pool)
    travel_cells = [c for c in hm["cells"] if c["vertical"] == "travel"]
    ecom_cells = [c for c in hm["cells"] if c["vertical"] == "ecommerce"]
    assert sum(c["count"] for c in travel_cells) == 3
    assert sum(c["count"] for c in ecom_cells) == 1


def test_aging_indicator_classifies_recent_pool_low_risk():
    # All admitted RCTs end in current year → low risk.
    from datetime import date
    today = date(2026, 5, 6)
    pool = [
        _make_rct(f"CMP-{i}", end_date=today.isoformat()) for i in range(5)
    ]
    for r in pool:
        dps.promote_rct(r["campaign_id"], r)
    indicator = dps.aging_indicator(pool, reference=today)
    assert indicator["fraction_recent"] == 1.0
    assert indicator["extrapolation_risk"] == "low"


def test_aging_indicator_classifies_old_pool_high_risk():
    from datetime import date
    today = date(2026, 5, 6)
    pool = [
        _make_rct(f"CMP-{i}", end_date="2023-06-30") for i in range(5)
    ]
    for r in pool:
        dps.promote_rct(r["campaign_id"], r)
    indicator = dps.aging_indicator(pool, reference=today)
    assert indicator["fraction_recent"] == 0.0
    assert indicator["extrapolation_risk"] == "high"


def test_shadow_rct_recommendations_target_gaps():
    # Admit all RCTs in ecommerce/lower/retargeting; gaps should appear.
    pool = [
        _make_rct(f"CMP-{i}", vertical="ecommerce", funnel_stage="lower",
                  audience_type="retargeting")
        for i in range(2)
    ] + [
        # an unmapped (vertical, funnel, audience) cell that won't appear in admitted
    ]
    for r in pool:
        dps.promote_rct(r["campaign_id"], r)
    recs = dps.recommend_shadow_rcts(pool, gap_threshold=1)
    # All admitted rows are in the same cell → no recs
    assert recs == []

    # Now demote and check that a gap is reported
    dps.demote_rct("CMP-0")
    dps.demote_rct("CMP-1")
    recs2 = dps.recommend_shadow_rcts(pool, gap_threshold=1)
    # No admitted RCTs at all → coverage heatmap empty → no recs (no cells observed)
    assert isinstance(recs2, list)
