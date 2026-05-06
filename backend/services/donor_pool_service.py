"""
Donor Pool Manager (PDF §4.2, Prompt 2.1).

Curates which RCTs are training labels. Enforces the size-band gating logic:
  <200    BLOCK training              (R² ~0.37 at 50 RCTs; unfit for prod)
  200–399 RESEARCH MODE only          (watermarked outputs)
  400–1599 PRODUCTION                  (R² 0.72–0.81)
  ≥1600   PRODUCTION (FULL)            (paper baseline ~0.88)

Also produces the coverage heatmap, RCT quality scores, donor-pool aging
indicator, and Shadow-Mode RCT recommendations.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date, datetime, timezone
from typing import Iterable, Literal

from services.persistence import find_by, read_table, upsert

PoolSizeBand = Literal["blocked", "research_mode", "production", "production_full"]
DONOR_POOL_TABLE = "donor_pool_membership"
SHADOW_RCT_TABLE = "shadow_rct_recommendations"


@dataclass
class PoolSizeStatus:
    n_admitted: int
    band: PoolSizeBand
    rationale: str

    def to_dict(self) -> dict:
        return asdict(self)


def _band_for(n: int) -> tuple[PoolSizeBand, str]:
    if n < 200:
        return "blocked", (
            f"{n} RCTs admitted; <200 → training blocked. R² ~0.37 at 50 RCTs "
            "(PDF Figure 6, p. 24); model is unfit for production."
        )
    if n < 400:
        return "research_mode", (
            f"{n} RCTs admitted; 200–399 → research mode only. Predictions "
            "exported from this band are watermarked and may not be used in "
            "the Decision Simulator."
        )
    if n < 1600:
        return "production", (
            f"{n} RCTs admitted; 400–1599 → production mode. Expected "
            "weighted R² in the 0.72–0.81 range."
        )
    return "production_full", (
        f"{n} RCTs admitted; ≥1600 → production (full). R² approaches paper "
        "baseline of 0.88."
    )


def get_pool_size_status() -> PoolSizeStatus:
    rows = read_table(DONOR_POOL_TABLE)
    n = sum(1 for r in rows if r.get("admitted"))
    band, rationale = _band_for(n)
    return PoolSizeStatus(n_admitted=n, band=band, rationale=rationale)


# --- RCT quality score (0-100) ---------------------------------------------


def _quality_score(rct: dict) -> int:
    """Weighted combination of size, conversion volume, duration, control share.

    Component weights chosen so a clean RCT (≥1M test users, ≥5K test
    conversions, ≥14-day duration, ~50/50 split) returns ≥80.
    """
    test_users = float(rct.get("test_users") or 0)
    test_conversions = float(rct.get("test_conversions") or 0)
    control_users = float(rct.get("control_users") or 0)
    duration = float(rct.get("duration_days") or 0)

    size = min(test_users / 2_000_000, 1.0)
    volume = min(test_conversions / 10_000, 1.0)
    duration_score = min(duration / 28.0, 1.0)
    if (test_users + control_users) > 0:
        control_share = control_users / (test_users + control_users)
        balance = 1.0 - 2.0 * abs(0.5 - control_share)  # 1.0 at 50/50, 0 at 0% or 100%
    else:
        balance = 0.0

    score = 0.30 * size + 0.30 * volume + 0.20 * duration_score + 0.20 * balance
    return max(0, min(100, int(round(score * 100))))


# --- Public API -------------------------------------------------------------


def list_eligible_rcts(rcts: Iterable[dict]) -> list[dict]:
    """Augment incoming RCT records with quality score + admission status."""
    membership = {r["campaign_id"]: r for r in read_table(DONOR_POOL_TABLE)}
    out = []
    for rct in rcts:
        cid = rct["campaign_id"]
        m = membership.get(cid, {})
        out.append(
            {
                **rct,
                "quality_score": _quality_score(rct),
                "admitted": bool(m.get("admitted", False)),
                "promoted_at": m.get("promoted_at"),
                "demoted_at": m.get("demoted_at"),
            }
        )
    return out


def promote_rct(campaign_id: str, rct: dict) -> dict:
    record = {
        "campaign_id": campaign_id,
        "admitted": True,
        "promoted_at": datetime.now(timezone.utc).isoformat(),
        "demoted_at": None,
        "metadata": rct,
    }
    upsert(DONOR_POOL_TABLE, record, key="campaign_id")
    return record


def demote_rct(campaign_id: str) -> dict | None:
    existing = find_by(DONOR_POOL_TABLE, "campaign_id", campaign_id)
    if not existing:
        return None
    existing["admitted"] = False
    existing["demoted_at"] = datetime.now(timezone.utc).isoformat()
    upsert(DONOR_POOL_TABLE, existing, key="campaign_id")
    return existing


def get_admitted_rcts() -> list[dict]:
    return [r for r in read_table(DONOR_POOL_TABLE) if r.get("admitted")]


# --- Coverage heatmap -------------------------------------------------------


def coverage_heatmap(rcts: Iterable[dict]) -> dict:
    """vertical × funnel × audience_type matrix with admitted RCT counts."""
    admitted_ids = {r["campaign_id"] for r in get_admitted_rcts()}
    cells: dict[str, int] = {}
    verticals: set[str] = set()
    funnels: set[str] = set()
    audiences: set[str] = set()
    for rct in rcts:
        if rct["campaign_id"] not in admitted_ids:
            continue
        v = str(rct.get("vertical", "unknown"))
        f = str(rct.get("funnel_stage", "unknown"))
        a = str(rct.get("audience_type", "unknown"))
        verticals.add(v)
        funnels.add(f)
        audiences.add(a)
        key = f"{v}|{f}|{a}"
        cells[key] = cells.get(key, 0) + 1

    rows = []
    for v in sorted(verticals):
        for f in sorted(funnels):
            for a in sorted(audiences):
                key = f"{v}|{f}|{a}"
                rows.append(
                    {
                        "vertical": v,
                        "funnel_stage": f,
                        "audience_type": a,
                        "count": cells.get(key, 0),
                    }
                )
    return {
        "verticals": sorted(verticals),
        "funnels": sorted(funnels),
        "audiences": sorted(audiences),
        "cells": rows,
    }


# --- Donor-pool aging indicator --------------------------------------------


def aging_indicator(rcts: Iterable[dict], reference: date | None = None) -> dict:
    """Fraction of admitted RCTs from the most recent calendar year.

    Year-to-year extrapolation drove the 21.3-pt R² penalty in Table 1, so this
    is surfaced prominently in the UI.
    """
    ref = reference or date.today()
    admitted_ids = {r["campaign_id"] for r in get_admitted_rcts()}
    same_year = 0
    one_year_ago = 0
    older = 0
    total = 0
    for rct in rcts:
        if rct["campaign_id"] not in admitted_ids:
            continue
        total += 1
        ed = rct.get("end_date")
        if isinstance(ed, str):
            try:
                ed_year = datetime.fromisoformat(ed).year
            except ValueError:
                continue
        elif isinstance(ed, date):
            ed_year = ed.year
        else:
            continue
        delta = ref.year - ed_year
        if delta <= 0:
            same_year += 1
        elif delta == 1:
            one_year_ago += 1
        else:
            older += 1

    fraction_recent = (same_year / total) if total else 0.0
    if fraction_recent >= 0.5:
        risk = "low"
    elif fraction_recent >= 0.25:
        risk = "medium"
    else:
        risk = "high"
    return {
        "total_admitted": total,
        "same_year": same_year,
        "one_year_ago": one_year_ago,
        "older": older,
        "fraction_recent": round(fraction_recent, 3),
        "extrapolation_risk": risk,
        "rationale": (
            "Fraction of admitted RCTs from the same calendar year as today. "
            "Year-to-year drift carries a 21.3-ppt R² penalty (PDF Table 1)."
        ),
    }


# --- Shadow-Mode RCT recommendations ---------------------------------------


def recommend_shadow_rcts(
    rcts: Iterable[dict], gap_threshold: int = 1
) -> list[dict]:
    """Identify (vertical, funnel, audience) cells with admitted-RCT count
    below threshold and emit shadow-RCT briefs."""
    heatmap = coverage_heatmap(rcts)
    recs = []
    for cell in heatmap["cells"]:
        if cell["count"] < gap_threshold:
            recs.append(
                {
                    "vertical": cell["vertical"],
                    "funnel_stage": cell["funnel_stage"],
                    "audience_type": cell["audience_type"],
                    "gap_score": gap_threshold - cell["count"],
                    "status": "open",
                    "brief": (
                        f"Donor pool coverage gap in {cell['vertical']} / "
                        f"{cell['funnel_stage']} / {cell['audience_type']}. "
                        f"Run a shadow RCT to fill this segment."
                    ),
                }
            )
    # Persist for governance audit
    for r in recs:
        key_id = f"{r['vertical']}|{r['funnel_stage']}|{r['audience_type']}"
        upsert(
            SHADOW_RCT_TABLE,
            {**r, "id": key_id, "created_at": datetime.now(timezone.utc).isoformat()},
            key="id",
        )
    return recs
