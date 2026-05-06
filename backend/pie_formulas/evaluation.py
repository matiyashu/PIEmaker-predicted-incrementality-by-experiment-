"""
Evaluation metrics (PDF Section 3.3).

Headline is cost-weighted out-of-sample R² (Eq. 24, p. 19). Bootstrap intervals
use 1,000 nonparametric draws by default. R² ceiling adjusts for the noise
floor in the ICPD label itself (Footnote 21, p. 19).
"""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np
from scipy import stats


def weighted_r_squared(
    y_true: Sequence[float],
    y_pred: Sequence[float],
    weights: Sequence[float],
) -> float:
    """Cost-weighted out-of-sample R-squared (HEADLINE METRIC).

    R² = 1 − Σ ω_r (ICPD_r − ICPD̂_r)² / Σ ω_r (ICPD_r − ICPD̄)²
    where ω_r = Cost_r / Σ Cost_r.

    Reference: Eq. 24, Gordon et al. 2026, p. 19.
    Larger campaigns are weighted more, matching ad-platform priorities.
    """
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    w = np.asarray(weights, dtype=float)
    if y_true_arr.shape != y_pred_arr.shape or y_true_arr.shape != w.shape:
        raise ValueError("y_true, y_pred, and weights must have the same shape")
    if y_true_arr.size == 0:
        raise ValueError("inputs must be non-empty")
    if np.any(w < 0):
        raise ValueError("weights must be non-negative")
    w_sum = w.sum()
    if w_sum <= 0:
        raise ValueError("sum of weights must be > 0")
    y_bar = (w * y_true_arr).sum() / w_sum
    numerator = (w * (y_true_arr - y_pred_arr) ** 2).sum()
    denominator = (w * (y_true_arr - y_bar) ** 2).sum()
    if denominator == 0:
        raise ValueError(
            "weighted variance of y_true is zero; R² undefined"
        )
    return 1.0 - (numerator / denominator)


def r_squared_ceiling(
    outcome_noise_variance: float, total_outcome_variance: float
) -> float:
    """Theoretical R² ceiling given label estimation noise.

    R²_ceiling = 1 − σ²_outcome_noise / σ²_total_outcome

    Reference: Footnote 21, Gordon et al. 2026, p. 19.
    A reported R² of 0.88 is closer to the ceiling than to 1.0; the Model Lab
    must surface both raw R² and the ceiling-adjusted view.
    """
    if total_outcome_variance <= 0:
        raise ValueError("total_outcome_variance must be > 0")
    if outcome_noise_variance < 0:
        raise ValueError("outcome_noise_variance must be >= 0")
    if outcome_noise_variance > total_outcome_variance:
        raise ValueError(
            "outcome_noise_variance cannot exceed total_outcome_variance"
        )
    return 1.0 - (outcome_noise_variance / total_outcome_variance)


def bootstrap_metric(
    metric_fn: Callable[..., float],
    *arrays: Sequence[float],
    n_draws: int = 1000,
    seed: int = 42,
) -> dict:
    """Nonparametric bootstrap distribution over RCT rows.

    Reference: Gordon et al. 2026, p. 19 (1,000-draw bootstrap).
    Resamples row indices with replacement, applies metric_fn to each draw, and
    returns mean/SD/percentiles. All input sequences must share the same length
    and align row-wise.
    """
    if n_draws < 1:
        raise ValueError("n_draws must be >= 1")
    if not arrays:
        raise ValueError("at least one array required")
    arrs = [np.asarray(a, dtype=float) for a in arrays]
    n = arrs[0].shape[0]
    if n == 0:
        raise ValueError("inputs must be non-empty")
    if any(a.shape[0] != n for a in arrs):
        raise ValueError("all input arrays must have the same length")
    rng = np.random.default_rng(seed)
    samples = np.empty(n_draws, dtype=float)
    for i in range(n_draws):
        idx = rng.integers(0, n, size=n)
        resampled = [a[idx] for a in arrs]
        samples[i] = metric_fn(*resampled)
    return {
        "mean": float(samples.mean()),
        "sd": float(samples.std(ddof=1)) if n_draws > 1 else 0.0,
        "p025": float(np.percentile(samples, 2.5)),
        "p975": float(np.percentile(samples, 97.5)),
        "n_draws": n_draws,
    }


def lcc_bias_ratio(lcc_per_dollar: Sequence[float], icpd: Sequence[float]) -> float:
    """LCC bias ratio (means): mean(LCC_7D / Cost) / mean(ICPD).

    Reference: Gordon et al. 2026, p. 23.
    Paper baseline: 1.33 overall; 1.2 ecomm, 1.5 retail, 2.0 travel.
    """
    lcc = np.asarray(lcc_per_dollar, dtype=float)
    truth = np.asarray(icpd, dtype=float)
    if lcc.size == 0 or truth.size == 0:
        raise ValueError("inputs must be non-empty")
    if lcc.shape != truth.shape:
        raise ValueError("inputs must have the same shape")
    icpd_mean = truth.mean()
    if icpd_mean == 0:
        raise ValueError("mean(ICPD) is zero; bias ratio undefined")
    return float(lcc.mean() / icpd_mean)


def lcc_ols_slope(
    icpd: Sequence[float], lcc_per_dollar: Sequence[float]
) -> float:
    """OLS slope of ICPD on (LCC_7D / Cost) — Must-have diagnostic.

    Reference: Gordon et al. 2026, p. 23.
    Paper baseline: 0.69 overall in Meta data — diagnostic of attribution
    overstatement.
    """
    y = np.asarray(icpd, dtype=float)
    x = np.asarray(lcc_per_dollar, dtype=float)
    if y.shape != x.shape:
        raise ValueError("inputs must have the same shape")
    if y.size < 2:
        raise ValueError("need at least 2 observations for OLS slope")
    if np.var(x) == 0:
        raise ValueError("LCC has zero variance; slope undefined")
    slope, _intercept, _r, _p, _se = stats.linregress(x, y)
    return float(slope)


def lcc_spearman_rho(
    icpd: Sequence[float], lcc_per_dollar: Sequence[float]
) -> float:
    """Spearman ρ between ICPD and LCC_7D / Cost — Must-have diagnostic.

    Reference: Gordon et al. 2026, p. 23.
    Paper baseline: 0.89 in Meta data — shows LCC carries salvageable signal
    even when biased in level.
    """
    y = np.asarray(icpd, dtype=float)
    x = np.asarray(lcc_per_dollar, dtype=float)
    if y.shape != x.shape:
        raise ValueError("inputs must have the same shape")
    if y.size < 2:
        raise ValueError("need at least 2 observations for Spearman ρ")
    rho, _p = stats.spearmanr(x, y)
    return float(rho)
