"""
Decision Simulator (Prompt 4.2 — final).

Interactive what-if for budget reallocation. Reallocates a fixed total
budget across non-blocked campaigns, proportional to their risk-adjusted
scores from decision_service, subject to a per-campaign cap multiplier
(e.g., cap=2.0 means no campaign's new budget can exceed 2× its original).

Hard-blocks research-mode models — the locked product rule says only
production-grade trust artifacts may drive monetary recommendations.

Math:
  IC_i      = predicted_icpd_i × spend_i        (incremental conversions)
  total_IC  = Σ IC_i

  weight_i  = max(0, risk_adjusted_score_i)     (block → weight 0)
  share_i   = weight_i / Σ weights              (uniform if all weights == 0)
  proposed_i = total_budget × share_i

Two-pass capping:
  1. cap each proposed_i at original_i × cap_multiplier
  2. redistribute the freed budget proportional to uncapped campaigns'
     weights (single pass; converges fast in practice for cap≥1.0)
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from services.decision_service import rank_recommendations
from services.prediction_service import (
    PredictionError,
    _select_model,
    score_portfolio,
)


class SimulatorError(Exception):
    """Raised when the simulator cannot run (research model, empty input, etc.)."""


@dataclass
class CampaignAllocation:
    run_id: str
    campaign_id: str
    action: str
    worst_risk: str
    predicted_icpd: float
    risk_adjusted_score: float | None
    original_spend: float
    proposed_spend: float
    delta_spend: float
    delta_pct: float | None
    capped: bool
    original_ic: float
    proposed_ic: float

    def to_dict(self) -> dict:
        return asdict(self)


def _spend_of(run: dict) -> float:
    """Pull the campaign's original spend out of its spec."""
    spec = run.get("spec") or {}
    raw = spec.get("cost")
    try:
        v = float(raw)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, v)


def _allocate(
    runs_with_recs: list[tuple[dict, dict]],
    *,
    total_budget: float,
    cap_multiplier: float,
) -> list[CampaignAllocation]:
    if total_budget <= 0:
        return []

    weights: list[float] = []
    originals: list[float] = []
    for run, rec in runs_with_recs:
        original = _spend_of(run)
        originals.append(original)
        if rec["action"] == "block":
            weights.append(0.0)
            continue
        score = rec["risk_adjusted_score"]
        w = max(0.0, float(score)) if score is not None else 0.0
        weights.append(w)

    sum_w = sum(weights)
    if sum_w <= 0:
        # All weights zero (e.g., everything blocked or score-less). Fall back
        # to equal split among non-blocked campaigns; if none, return zeros.
        non_blocked_idx = [
            i for i, (_, rec) in enumerate(runs_with_recs)
            if rec["action"] != "block"
        ]
        if not non_blocked_idx:
            proposals = [0.0] * len(runs_with_recs)
        else:
            even = total_budget / len(non_blocked_idx)
            proposals = [
                even if i in non_blocked_idx else 0.0
                for i in range(len(runs_with_recs))
            ]
    else:
        proposals = [total_budget * (w / sum_w) for w in weights]

    # Iteratively cap-and-redistribute. Each pass caps any proposal that
    # exceeds original * cap_multiplier, then redistributes the freed
    # budget to still-uncapped campaigns proportional to their weights.
    # Loop converges when no excess is freed in a pass, or after a small
    # iteration limit (max_iters protects against pathological inputs).
    caps = [orig * cap_multiplier for orig in originals]
    capped_flags = [False] * len(proposals)
    max_iters = 8

    for _ in range(max_iters):
        excess = 0.0
        for i, p in enumerate(proposals):
            c = caps[i]
            if not capped_flags[i] and c > 0 and p > c:
                excess += p - c
                proposals[i] = c
                capped_flags[i] = True
        if excess <= 1e-9:
            break
        uncapped = [
            i
            for i, _ in enumerate(proposals)
            if not capped_flags[i]
            and runs_with_recs[i][1]["action"] != "block"
        ]
        if not uncapped:
            break
        sub_w = sum(weights[i] for i in uncapped)
        if sub_w > 0:
            for i in uncapped:
                share = weights[i] / sub_w
                proposals[i] += excess * share
        else:
            for i in uncapped:
                proposals[i] += excess / len(uncapped)

    out: list[CampaignAllocation] = []
    for (run, rec), original, proposed, capped in zip(
        runs_with_recs, originals, proposals, capped_flags, strict=True
    ):
        delta = proposed - original
        delta_pct = (delta / original * 100.0) if original > 0 else None
        icpd = float(run["predicted_icpd"])
        out.append(
            CampaignAllocation(
                run_id=run["id"],
                campaign_id=run["campaign_id"],
                action=rec["action"],
                worst_risk=rec["worst_risk"],
                predicted_icpd=icpd,
                risk_adjusted_score=rec.get("risk_adjusted_score"),
                original_spend=round(original, 4),
                proposed_spend=round(proposed, 4),
                delta_spend=round(delta, 4),
                delta_pct=round(delta_pct, 2) if delta_pct is not None else None,
                capped=capped,
                original_ic=round(icpd * original, 4),
                proposed_ic=round(icpd * proposed, 4),
            )
        )
    return out


def simulate(
    rows: list[dict],
    *,
    model_id: str | None = None,
    feature_set_version: str = "v1",
    cap_multiplier: float = 2.0,
    total_budget_override: float | None = None,
    risk_floor: str = "low",
) -> dict:
    """Run the simulator on a list of campaign specs.

    Hard-blocks research-mode models. Picks the model once (delegates to
    score_portfolio); falls back to the same auto-selection rule.
    """
    if not rows:
        raise SimulatorError("rows must not be empty")

    # Pre-select the model so we can reject research-mode before paying for
    # the feature build + prediction batch.
    try:
        candidate = _select_model(model_id)
    except PredictionError as exc:
        raise SimulatorError(str(exc))
    if candidate.get("status") == "research":
        raise SimulatorError(
            "The Decision Simulator is gated to production-grade models "
            "only (donor pool ≥ 400 RCTs). The selected model is in "
            "research mode. Promote it on the Model Trust page once the "
            "pool has enough labeled RCTs, or pick another production model."
        )

    portfolio = score_portfolio(
        rows,
        model_id=candidate["id"],
        feature_set_version=feature_set_version,
    )
    runs = portfolio["runs"]
    recs = rank_recommendations(runs, risk_floor=risk_floor)  # type: ignore[arg-type]
    rec_by_run = {r["run_id"]: r for r in recs}
    runs_with_recs = [(r, rec_by_run[r["id"]]) for r in runs]

    original_total = sum(_spend_of(r) for r in runs)
    total_budget = (
        float(total_budget_override)
        if total_budget_override is not None
        else original_total
    )

    allocations = _allocate(
        runs_with_recs,
        total_budget=total_budget,
        cap_multiplier=cap_multiplier,
    )

    original_ic_total = sum(a.original_ic for a in allocations)
    proposed_ic_total = sum(a.proposed_ic for a in allocations)
    ic_lift_pct = (
        ((proposed_ic_total - original_ic_total) / original_ic_total) * 100.0
        if original_ic_total > 0
        else None
    )

    n_capped = sum(1 for a in allocations if a.capped)
    n_blocked = sum(1 for a in allocations if a.action == "block")
    n_promoted = sum(1 for a in allocations if a.action == "promote")

    return {
        "model": portfolio["model"],
        "cap_multiplier": cap_multiplier,
        "risk_floor": risk_floor,
        "original_total_budget": round(original_total, 4),
        "total_budget": round(total_budget, 4),
        "original_ic_total": round(original_ic_total, 4),
        "proposed_ic_total": round(proposed_ic_total, 4),
        "ic_lift_pct": round(ic_lift_pct, 4) if ic_lift_pct is not None else None,
        "n_campaigns": len(allocations),
        "n_blocked": n_blocked,
        "n_capped": n_capped,
        "n_promoted": n_promoted,
        "allocations": [a.to_dict() for a in allocations],
        "rationale": (
            f"Reallocated ${total_budget:,.2f} across {len(allocations)} "
            f"campaigns ({n_blocked} blocked) using risk-adjusted weights "
            f"with a cap of {cap_multiplier:g}× original spend per campaign. "
            f"{n_capped} campaign(s) hit the cap; freed budget redistributed "
            f"proportional to remaining weights."
        ),
    }
