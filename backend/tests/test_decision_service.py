"""Decision recommendations tests (Prompt 3.3)."""

from __future__ import annotations

from services.decision_service import (
    project_portfolio_lift,
    rank_recommendations,
)


def _run(
    rid: str,
    cid: str,
    icpd: float,
    risk: str | None = None,
    ci_lower: float | None = None,
    ci_upper: float | None = None,
    model_status: str = "production",
) -> dict:
    return {
        "id": rid,
        "campaign_id": cid,
        "predicted_icpd": icpd,
        "ci_lower": ci_lower if ci_lower is not None else icpd - 0.05,
        "ci_upper": ci_upper if ci_upper is not None else icpd + 0.05,
        "model_status": model_status,
        "worst_segment_risk": (
            {"risk": risk, "segmentation_var": "vertical", "level": "x", "penalty_pp": 1.0}
            if risk
            else None
        ),
    }


def test_severe_risk_is_blocked_and_sinks_to_bottom():
    runs = [
        _run("a", "CMP-A", 0.5, risk="severe"),
        _run("b", "CMP-B", 0.3, risk="low"),
    ]
    recs = rank_recommendations(runs)
    last = recs[-1]
    assert last["campaign_id"] == "CMP-A"
    assert last["action"] == "block"


def test_negative_lower_ci_deprioritized():
    runs = [
        _run("a", "CMP-A", 0.10, ci_lower=-0.05, ci_upper=0.25, risk="low"),
        _run("b", "CMP-B", 0.40, ci_lower=0.30, ci_upper=0.50, risk="low"),
    ]
    recs = rank_recommendations(runs)
    by_cid = {r["campaign_id"]: r for r in recs}
    assert by_cid["CMP-A"]["action"] == "deprioritize"


def test_high_risk_deprioritized():
    runs = [_run("a", "CMP-A", 0.5, risk="high")]
    recs = rank_recommendations(runs)
    assert recs[0]["action"] == "deprioritize"


def test_medium_risk_held():
    runs = [_run("a", "CMP-A", 0.5, risk="medium")]
    recs = rank_recommendations(runs)
    assert recs[0]["action"] == "hold"


def test_top_tercile_low_risk_promoted():
    runs = [
        _run(f"r{i}", f"CMP-{i}", 0.10 + i * 0.05, risk="low") for i in range(9)
    ]
    recs = rank_recommendations(runs)
    promotes = [r for r in recs if r["action"] == "promote"]
    # Top 1/3 of 9 = 3 candidates above the 67th percentile
    assert len(promotes) >= 1
    # The single highest-ICPD row must be promoted
    assert recs[0]["action"] == "promote"
    assert recs[0]["campaign_id"] == "CMP-8"


def test_ranking_uses_ci_lower_when_present():
    """A high-mean / wide-CI campaign should rank below a moderate-mean /
    tight-CI campaign once the CI floor is applied."""
    runs = [
        _run("wide", "CMP-WIDE", 1.0, ci_lower=0.10, ci_upper=1.90, risk="low"),
        _run("tight", "CMP-TIGHT", 0.50, ci_lower=0.45, ci_upper=0.55, risk="low"),
    ]
    recs = rank_recommendations(runs)
    assert recs[0]["campaign_id"] == "CMP-TIGHT"


def test_risk_floor_downgrades_promote_to_hold():
    runs = [
        _run("r1", "CMP-A", 0.50, risk="medium"),
        _run("r2", "CMP-B", 0.30, risk="medium"),
        _run("r3", "CMP-C", 0.20, risk="medium"),
    ]
    # With risk_floor='low', no medium-risk row can be 'promote'
    recs = rank_recommendations(runs, risk_floor="low")
    assert all(r["action"] != "promote" for r in recs)


def test_projected_lift_excludes_blocked_and_deprioritized():
    runs = [
        _run("a", "CMP-A", 0.10, risk="severe"),       # block
        _run("b", "CMP-B", 0.20, risk="high"),          # deprioritize
        _run("c", "CMP-C", 0.50, risk="low"),           # promote
        _run("d", "CMP-D", 0.40, risk="low"),           # promote-or-hold
    ]
    recs = rank_recommendations(runs)
    lift = project_portfolio_lift(recs, is_research_model=False)
    assert lift is not None
    # Followed mean should exclude C/D's risky peers
    assert lift["n_total"] == 4
    assert lift["n_followed"] >= 1
    assert lift["advised_portfolio_icpd"] >= lift["naive_portfolio_icpd"]


def test_projected_lift_none_for_research_model():
    runs = [_run("a", "CMP-A", 0.5, risk="low")]
    recs = rank_recommendations(runs)
    assert project_portfolio_lift(recs, is_research_model=True) is None


def test_empty_input_returns_empty_recs():
    assert rank_recommendations([]) == []


def test_rank_indices_are_contiguous_starting_at_one():
    runs = [_run(f"r{i}", f"CMP-{i}", 0.1 * i, risk="low") for i in range(5)]
    recs = rank_recommendations(runs)
    assert [r["rank"] for r in recs] == [1, 2, 3, 4, 5]
