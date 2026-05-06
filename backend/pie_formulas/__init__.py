"""
PIE Formula Contracts — FROZEN MODULE.

Source: Gordon, Moakler & Zettelmeyer (NBER w35044, April 2026).

These functions are frozen contracts. Any change requires a documented design
review and a version bump in the model registry. The CI workflow at
.github/workflows/formula-contracts.yml fails any PR that breaks the contract
tests in backend/tests/test_pie_formulas.py.

Equation and page references throughout cite the source paper.
"""

from pie_formulas.labels import att, exposure_rate, icpd, incremental_conversions
from pie_formulas.prediction import (
    incremental_revenue,
    iroas,
    predicted_cpic,
    predicted_ic,
)
from pie_formulas.evaluation import (
    bootstrap_metric,
    lcc_bias_ratio,
    lcc_ols_slope,
    lcc_spearman_rho,
    r_squared_ceiling,
    weighted_r_squared,
)
from pie_formulas.decision import (
    disagreement_probability,
    expected_disagreement_cost,
    segment_median_threshold,
    threshold_scan_range,
)
from pie_formulas.research import att_decomposition

__all__ = [
    "att",
    "att_decomposition",
    "bootstrap_metric",
    "disagreement_probability",
    "expected_disagreement_cost",
    "exposure_rate",
    "icpd",
    "incremental_conversions",
    "incremental_revenue",
    "iroas",
    "lcc_bias_ratio",
    "lcc_ols_slope",
    "lcc_spearman_rho",
    "predicted_cpic",
    "predicted_ic",
    "r_squared_ceiling",
    "segment_median_threshold",
    "threshold_scan_range",
    "weighted_r_squared",
]

__version__ = "0.1.0"
