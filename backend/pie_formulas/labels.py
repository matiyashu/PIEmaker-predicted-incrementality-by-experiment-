"""
Label generation formulas (PDF Section 3.1).

Generates ground-truth ATT, IC, and ICPD per RCT for use as PIE training labels.
"""

from __future__ import annotations


def exposure_rate(exposed_test_users: float, test_users: float) -> float:
    """Exposure rate D̄_tr = exposed_test_users / test_users.

    Reference: Gordon et al. 2026, p. 11.
    Required as the ATT denominator for one-sided noncompliance.

    Raises:
        ValueError: if test_users <= 0 or exposed_test_users < 0
            or exposure rate falls outside (0, 1].
    """
    if test_users <= 0:
        raise ValueError("test_users must be > 0")
    if exposed_test_users < 0:
        raise ValueError("exposed_test_users must be >= 0")
    rate = exposed_test_users / test_users
    if not (0 < rate <= 1):
        raise ValueError(
            f"exposure rate must satisfy 0 < D̄_tr <= 1, got {rate}"
        )
    return rate


def att(
    test_conversions: float,
    test_users: float,
    control_conversions: float,
    control_users: float,
    exposure_rate_value: float,
) -> float:
    """Imbens-Angrist Wald/2SLS ATT for one-sided noncompliance.

    ψ̂_r = (Ȳ_tr − Ȳ_cr) / D̄_tr

    Reference: Eq. 14, Gordon et al. 2026, p. 11.
    """
    if test_users <= 0 or control_users <= 0:
        raise ValueError("test_users and control_users must be > 0")
    if not (0 < exposure_rate_value <= 1):
        raise ValueError("exposure_rate_value must satisfy 0 < D̄_tr <= 1")
    y_tr = test_conversions / test_users
    y_cr = control_conversions / control_users
    return (y_tr - y_cr) / exposure_rate_value


def incremental_conversions(
    att_value: float, exposure_rate_value: float, test_users: float
) -> float:
    """Incremental Conversions IC_r = ATT × D_tr × N_tr.

    Reference: Eq. 22, Gordon et al. 2026, p. 18.
    D_tr × N_tr equals exposed test users.
    """
    if test_users <= 0:
        raise ValueError("test_users must be > 0")
    if not (0 < exposure_rate_value <= 1):
        raise ValueError("exposure_rate_value must satisfy 0 < D̄_tr <= 1")
    return att_value * exposure_rate_value * test_users


def icpd(incremental_conversions_value: float, cost: float) -> float:
    """Incremental Conversions Per Dollar — headline KPI.

    ICPD_r = IC_r / Cost_r

    Reference: Eq. 23, Gordon et al. 2026, p. 18.
    Chosen over CPIC because ICPD handles zero/negative lift cleanly.
    """
    if cost <= 0:
        raise ValueError("cost must be > 0 (ICPD denominator)")
    return incremental_conversions_value / cost
