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

    Deprecated in V.4 in favour of ``r_squared_ceiling_from_label_noise``,
    which derives σ²_label from per-RCT ATT standard errors (paper-faithful)
    rather than the model's residual variance (which conflates label noise
    with model misspecification). Kept for back-compat.
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


def att_variance(
    test_conversions: float,
    test_users: float,
    control_conversions: float,
    control_users: float,
    exposure_rate_value: float,
) -> float:
    """Variance of the ATT label estimator for one RCT.

    Treats the per-arm conversion rates as binomial proportions:
        Var(p_t) = p_t (1 − p_t) / n_t
        Var(p_c) = p_c (1 − p_c) / n_c
        Var(p_t − p_c) = Var(p_t) + Var(p_c)
        Var(ATT) = Var(p_t − p_c) / D_bar²

    Paper §3.1 Footnote 21: this Bernoulli-difference-of-proportions form is
    the per-RCT label noise that drives the R² ceiling.
    """
    if test_users <= 0 or control_users <= 0:
        raise ValueError("test_users and control_users must be > 0")
    if exposure_rate_value <= 0:
        raise ValueError("exposure_rate_value must be > 0")
    p_t = test_conversions / test_users
    p_c = control_conversions / control_users
    var_diff = (p_t * (1.0 - p_t) / test_users) + (
        p_c * (1.0 - p_c) / control_users
    )
    return var_diff / (exposure_rate_value ** 2)


def icpd_label_variance(
    test_conversions: float,
    test_users: float,
    control_conversions: float,
    control_users: float,
    exposure_rate_value: float,
    cost: float,
) -> float:
    """Variance of the ICPD label for one RCT, propagating ATT noise.

    ICPD = ATT × D_bar × test_users / cost  (Eqs. 14 → 22 → 23)
    The deterministic scaling factor `(D_bar × test_users / cost)` lifts
    Var(ATT) to Var(ICPD): Var(ICPD) = (D_bar × test_users / cost)² × Var(ATT).
    """
    if cost <= 0:
        raise ValueError("cost must be > 0")
    scale = (exposure_rate_value * test_users) / cost
    return (scale ** 2) * att_variance(
        test_conversions,
        test_users,
        control_conversions,
        control_users,
        exposure_rate_value,
    )


def r_squared_ceiling_from_label_noise(
    icpd_label_variances: Sequence[float],
    icpd_labels: Sequence[float],
    weights: Sequence[float] | None = None,
) -> float:
    """Paper-faithful R² ceiling from per-RCT label-noise variance.

    σ²_label_aggregate = Σ w_i σ²_label_i / Σ w_i
    σ²_total = weighted sample variance of ICPD labels
    R²_ceiling = 1 − σ²_label_aggregate / σ²_total

    Paper §5.2: this isolates label noise from model error, so a reported R²
    that approaches the ceiling means the model is at the noise floor (more
    features cannot help) — distinct from a model that is simply underfit.
    """
    if not icpd_label_variances:
        raise ValueError("icpd_label_variances must be non-empty")
    if len(icpd_label_variances) != len(icpd_labels):
        raise ValueError(
            "icpd_label_variances and icpd_labels must have the same length"
        )
    w = (
        np.asarray(weights, dtype=float)
        if weights is not None
        else np.ones(len(icpd_label_variances), dtype=float)
    )
    if w.sum() <= 0:
        raise ValueError("weights must sum to > 0")
    var_arr = np.asarray(icpd_label_variances, dtype=float)
    y = np.asarray(icpd_labels, dtype=float)

    sigma2_label = float(np.sum(w * var_arr) / np.sum(w))
    mean_w = float(np.sum(w * y) / np.sum(w))
    sigma2_total = float(np.sum(w * (y - mean_w) ** 2) / np.sum(w))

    if sigma2_total <= 0:
        # Degenerate (all labels identical) — undefined ceiling; report 1.0.
        return 1.0
    # Clamp to [0, 1]: numerical noise or per-row variance overestimates
    # (e.g. when test/control rates are at the binomial bound) can push
    # the raw ratio outside the unit interval. The ceiling is a bound,
    # not a free parameter, so clamping is the right thing.
    ratio = sigma2_label / sigma2_total
    return float(max(0.0, min(1.0, 1.0 - ratio)))


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
