"""
Prediction formulas (PDF Section 3.2).

Convert model output (predicted ICPD) into business KPIs. CPIC is hard-blocked
when predicted ICPD <= 0 because the inverse is undefined or unstable.
"""

from __future__ import annotations

from typing import Optional


def predicted_ic(predicted_icpd: float, cost: float) -> float:
    """Predicted Incremental Conversions IĈ_r = ICPD̂_r × Cost_r.

    Reference: Inverse of Eq. 23, Gordon et al. 2026, p. 18.
    """
    if cost <= 0:
        raise ValueError("cost must be > 0")
    return predicted_icpd * cost


def predicted_cpic(predicted_icpd: float) -> Optional[float]:
    """Predicted CPIC = 1 / ICPD̂_r — HARD-BLOCKED when ICPD̂ <= 0.

    Reference: Footnote 18, Gordon et al. 2026, p. 18.
    Returns None when the predicted ICPD is non-positive; callers must surface
    the explanatory message "CPIC undefined: predicted incrementality is zero
    or negative."
    """
    if predicted_icpd <= 0:
        return None
    return 1.0 / predicted_icpd


def incremental_revenue(
    predicted_ic_value: float, revenue_per_conversion: float
) -> float:
    """Predicted Incremental Revenue IR_r = IĈ_r × Revenue_per_Conversion.

    Reference: Standard derivation; PDF §3.2.
    Requires the conversion_value field to be present.
    """
    if revenue_per_conversion < 0:
        raise ValueError("revenue_per_conversion must be >= 0")
    return predicted_ic_value * revenue_per_conversion


def iroas(predicted_incremental_revenue: float, cost: float) -> float:
    """Predicted Incremental ROAS iROAS_r = IR_r / Cost_r.

    Reference: Standard derivation; PDF §3.2.
    Business-facing metric.
    """
    if cost <= 0:
        raise ValueError("cost must be > 0")
    return predicted_incremental_revenue / cost
