"""
Decision Recommendations (Prompt 3.3).

Passive ranker for a portfolio of predictions. Tags each campaign with an
advisory action band based on the (predicted_icpd, ci_lower, worst-segment
risk) tuple, then ranks by a risk-adjusted score. Does NOT mutate budget —
that's the Phase 4.2 Decision Simulator's job. This module only proposes.

Action bands:
  block          severe extrapolation risk  (paper Table 1: ≥25pp penalty)
  deprioritize   high extrapolation risk    OR ci_lower ≤ 0
  hold           medium risk OR low-but-mediocre ICPD
  promote        low/unknown risk AND ICPD in the top tercile

Risk-adjusted score:
  score = ci_lower if available, else predicted_icpd
  risk_adjustment: severe → drop to -inf (sorts last), high → score * 0.5,
  medium → score * 0.85, low/unknown → score * 1.0.

Watermark/research-mode gating: research models still produce
recommendations, but every action is downgraded to "advisory_only" and the
projected_portfolio_icpd is omitted. The Decision Simulator (4.2) will hard-
block research-mode runs entirely.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import asdict, dataclass
from typing import Literal

ActionBand = Literal["block", "deprioritize", "hold", "promote"]
RiskLevel = Literal["severe", "high", "medium", "low", "unknown"]

_RISK_MULTIPLIER: dict[str, float] = {
    "severe": float("-inf"),
    "high": 0.5,
    "medium": 0.85,
    "low": 1.0,
    "unknown": 1.0,
}


@dataclass
class Recommendation:
    run_id: str
    campaign_id: str
    predicted_icpd: float
    ci_lower: float | None
    ci_upper: float | None
    worst_risk: RiskLevel
    risk_adjusted_score: float
    action: ActionBand
    rationale: str
    rank: int

    def to_dict(self) -> dict:
        return asdict(self)


def _score_one(run: dict) -> tuple[float, RiskLevel]:
    risk: RiskLevel = (
        run.get("worst_segment_risk", {}).get("risk", "unknown")
        if run.get("worst_segment_risk")
        else "unknown"
    )
    base = (
        run["ci_lower"]
        if run.get("ci_lower") is not None
        else run["predicted_icpd"]
    )
    mult = _RISK_MULTIPLIER.get(risk, 1.0)
    if math.isinf(mult):
        return (mult, risk)
    return (base * mult, risk)


def _action_band(
    run: dict, score: float, risk: RiskLevel, top_tercile_threshold: float
) -> tuple[ActionBand, str]:
    ci_lower = run.get("ci_lower")
    if risk == "severe":
        return ("block", "Severe extrapolation risk on a segment level — "
                          "training pool does not cover this campaign's regime.")
    if ci_lower is not None and ci_lower <= 0.0:
        return ("deprioritize", "Lower 95% CI bound is ≤ 0 — predicted ICPD "
                                "is not significantly above zero.")
    if risk == "high":
        return ("deprioritize", "High extrapolation risk; predictions on this "
                                "level have shown ≥15pp R² penalty.")
    if risk == "medium":
        return ("hold", "Medium extrapolation risk (≥5pp penalty). Run a "
                        "shadow RCT in this segment to upgrade trust.")
    # risk in {low, unknown}
    if score >= top_tercile_threshold:
        return ("promote", "Top-tercile risk-adjusted ICPD with low/unknown "
                           "extrapolation risk.")
    return ("hold", "Below top-tercile risk-adjusted ICPD; not promote-worthy "
                    "but no risk flag fires.")


def rank_recommendations(
    runs: list[dict],
    *,
    risk_floor: RiskLevel = "low",
) -> list[dict]:
    """Rank prediction_runs and tag each with an advisory action.

    `runs` should be the `runs` list from a portfolio response (or the
    output of list_runs()). `risk_floor` upgrades the action band — any
    campaign with risk *above* the floor cannot be 'promote'.
    """
    if not runs:
        return []

    # First pass: scores
    scored: list[tuple[dict, float, RiskLevel]] = []
    for run in runs:
        score, risk = _score_one(run)
        scored.append((run, score, risk))

    # Top-tercile threshold computed across promotable runs only
    promotable_scores = [
        s for _, s, r in scored if not math.isinf(s) and r in ("low", "unknown")
    ]
    if promotable_scores:
        # 67th percentile
        try:
            top_tercile = statistics.quantiles(promotable_scores, n=3)[-1]
        except statistics.StatisticsError:
            top_tercile = max(promotable_scores)
    else:
        top_tercile = float("inf")

    # Sort by risk-adjusted score desc; -inf (severe) sinks to bottom
    scored.sort(key=lambda t: t[1], reverse=True)

    out: list[Recommendation] = []
    for rank_idx, (run, score, risk) in enumerate(scored, start=1):
        action, rationale = _action_band(run, score, risk, top_tercile)
        # Apply risk_floor: anything above the floor can't be 'promote'
        floor_order = ["low", "unknown", "medium", "high", "severe"]
        if action == "promote" and floor_order.index(risk) > floor_order.index(
            risk_floor
        ):
            action = "hold"
            rationale = (
                f"Risk floor set to '{risk_floor}'; this campaign's risk "
                f"'{risk}' exceeds the floor — downgraded from promote to hold."
            )
        out.append(
            Recommendation(
                run_id=run["id"],
                campaign_id=run["campaign_id"],
                predicted_icpd=run["predicted_icpd"],
                ci_lower=run.get("ci_lower"),
                ci_upper=run.get("ci_upper"),
                worst_risk=risk,
                risk_adjusted_score=(
                    None if math.isinf(score) else round(score, 6)  # type: ignore[arg-type]
                ),
                action=action,
                rationale=rationale,
                rank=rank_idx,
            )
        )
    return [r.to_dict() for r in out]


def project_portfolio_lift(
    recommendations: list[dict],
    *,
    is_research_model: bool,
) -> dict | None:
    """Project portfolio mean ICPD if the user follows the recs.

    Implementation is intentionally simple and advisory:
      naive_mean   = mean(predicted_icpd)  across all runs
      followed_mean = mean(predicted_icpd) across {promote, hold} only
    Research-mode runs return None — the simulator is the only legitimate
    surface for actionable lift numbers.
    """
    if is_research_model or not recommendations:
        return None

    icpds = [r["predicted_icpd"] for r in recommendations]
    followed = [
        r["predicted_icpd"]
        for r in recommendations
        if r["action"] in ("promote", "hold")
    ]
    naive = float(statistics.mean(icpds)) if icpds else 0.0
    followed_mean = (
        float(statistics.mean(followed)) if followed else 0.0
    )
    return {
        "naive_portfolio_icpd": round(naive, 6),
        "advised_portfolio_icpd": round(followed_mean, 6),
        "lift_pp": round((followed_mean - naive) * 100.0, 4),
        "n_followed": len(followed),
        "n_total": len(recommendations),
        "rationale": (
            "Naive mean ICPD across all candidates vs. the mean across the "
            "campaigns the recommender keeps (promote + hold). Block and "
            "deprioritize are excluded. Phase 4.2's Decision Simulator will "
            "extend this with explicit budget redistribution."
        ),
    }
