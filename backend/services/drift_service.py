"""
Drift Monitoring (Prompt 4.1).

Population Stability Index (PSI) per feature, comparing the scoring
distribution to the training feature_store. PSI is the standard industry
choice for tabular feature drift — interpretable, scale-invariant, and a
single number per feature.

For each feature column f:
  PSI(f) = Σ_bin (a_b - e_b) × ln(a_b / e_b)
  where a_b is the actual share in scoring, e_b the expected share in training.
  Both shares are smoothed by ε = 1e-6 to avoid log(0) and divide-by-zero.

Numeric features: 10 quantile-based bins computed from the training
distribution. Categoricals: each observed level is its own bin (union of
training + scoring levels, smoothed).

Severity bands (industry standard):
  PSI < 0.10        stable
  0.10 ≤ PSI < 0.25 moderate
  PSI ≥ 0.25        severe

Verdict logic for the portfolio as a whole:
  any severe drifters → "retrain_recommended"
  ≥3 moderate drifters → "watch"
  else                 → "stable"
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Iterable, Literal

import numpy as np

from services.feature_engineering_service import (
    FEATURE_TABLE,
    X_POST_FIELDS,
    X_PRE_FIELDS,
)
from services.persistence import read_table

DriftSeverity = Literal["stable", "moderate", "severe"]
Verdict = Literal["stable", "watch", "retrain_recommended"]

_EPS = 1e-6
_MODERATE = 0.10
_SEVERE = 0.25


@dataclass
class FeatureDrift:
    feature: str
    kind: Literal["numeric", "categorical"]
    psi: float
    severity: DriftSeverity
    n_train: int
    n_score: int
    bins: list[dict]  # diagnostic detail per bin

    def to_dict(self) -> dict:
        return asdict(self)


def _classify(psi: float) -> DriftSeverity:
    if psi < _MODERATE:
        return "stable"
    if psi < _SEVERE:
        return "moderate"
    return "severe"


def _flatten_feature_rows(rows: Iterable[dict]) -> dict[str, list]:
    """Pivot list-of-rows into column-of-values, pulling x_pre/x_post nested dicts."""
    cols: dict[str, list] = {}
    for r in rows:
        for sub in ("x_pre", "x_post"):
            inner = r.get(sub) or {}
            for k, v in inner.items():
                cols.setdefault(k, []).append(v)
    return cols


def _is_numeric_feature(name: str) -> bool:
    """Treat the trainer's known numeric columns as numeric; everything else
    in the X_pre/X_post universe as categorical."""
    numeric_pre = {
        "advertiser_platform_experience_months",
        "month",
        "quarter",
        "campaign_duration_days",
    }
    numeric_post = set(X_POST_FIELDS)
    return name in numeric_pre or name in numeric_post


def _psi_numeric(
    train: list, score: list
) -> tuple[float, list[dict]]:
    train_arr = np.array(
        [float(v) for v in train if v is not None and not _is_nan(v)],
        dtype=float,
    )
    score_arr = np.array(
        [float(v) for v in score if v is not None and not _is_nan(v)],
        dtype=float,
    )
    if len(train_arr) < 10 or len(score_arr) == 0:
        return (0.0, [])

    quantiles = np.quantile(train_arr, np.linspace(0, 1, 11))
    quantiles = np.unique(quantiles)
    if len(quantiles) < 3:
        # Constant-ish training feature; report 0 drift unless scoring has values
        # outside the constant.
        same = np.allclose(score_arr, train_arr.mean())
        return (0.0, []) if same else (
            float("inf"),
            [{"bin": "out_of_range", "expected": 1.0, "actual": 1.0}],
        )

    train_counts, _ = np.histogram(train_arr, bins=quantiles)
    score_counts, _ = np.histogram(score_arr, bins=quantiles)
    expected = train_counts / max(1, train_counts.sum())
    actual = score_counts / max(1, score_counts.sum())

    e = np.clip(expected, _EPS, None)
    a = np.clip(actual, _EPS, None)
    psi = float(np.sum((a - e) * np.log(a / e)))

    bins_detail = [
        {
            "bin": f"[{quantiles[i]:.4g}, {quantiles[i + 1]:.4g})",
            "expected": float(e[i]),
            "actual": float(a[i]),
            "delta": float(a[i] - e[i]),
        }
        for i in range(len(e))
    ]
    return (psi, bins_detail)


def _psi_categorical(
    train: list, score: list
) -> tuple[float, list[dict]]:
    train_clean = [str(v) for v in train if v is not None and not _is_nan(v)]
    score_clean = [str(v) for v in score if v is not None and not _is_nan(v)]
    if len(train_clean) == 0 or len(score_clean) == 0:
        return (0.0, [])

    levels = sorted(set(train_clean) | set(score_clean))
    n_t = len(train_clean)
    n_s = len(score_clean)
    bins_detail: list[dict] = []
    psi = 0.0
    for lvl in levels:
        e = train_clean.count(lvl) / n_t
        a = score_clean.count(lvl) / n_s
        e_s = max(e, _EPS)
        a_s = max(a, _EPS)
        contribution = (a_s - e_s) * math.log(a_s / e_s)
        psi += contribution
        bins_detail.append(
            {
                "bin": lvl,
                "expected": e,
                "actual": a,
                "delta": a - e,
            }
        )
    return (float(psi), bins_detail)


def _is_nan(v) -> bool:
    try:
        return math.isnan(float(v))
    except (TypeError, ValueError):
        return False


def compute_drift(
    scoring_rows: list[dict],
    *,
    feature_set_version: str = "v1",
) -> dict:
    """Compare scoring features (already in feature_store row shape) against
    the training feature_store rows for the given version."""
    if not scoring_rows:
        raise ValueError("scoring_rows must not be empty")

    training_rows = [
        r
        for r in read_table(FEATURE_TABLE)
        if r.get("feature_set_version") == feature_set_version
        and r.get("mode") == "training"
    ]
    if not training_rows:
        raise ValueError(
            f"no training feature_store rows for version '{feature_set_version}' — "
            "build features in training mode first"
        )

    train_cols = _flatten_feature_rows(training_rows)
    score_cols = _flatten_feature_rows(scoring_rows)

    drifts: list[FeatureDrift] = []
    for feature in (*X_PRE_FIELDS, *X_POST_FIELDS):
        train_vals = train_cols.get(feature, [])
        score_vals = score_cols.get(feature, [])
        if not train_vals or not score_vals:
            continue
        if _is_numeric_feature(feature):
            psi, bins = _psi_numeric(train_vals, score_vals)
            kind = "numeric"
        else:
            psi, bins = _psi_categorical(train_vals, score_vals)
            kind = "categorical"
        if not math.isfinite(psi):
            psi = float(_SEVERE * 4)  # cap so JSON-serializable; still 'severe'
        drifts.append(
            FeatureDrift(
                feature=feature,
                kind=kind,
                psi=round(psi, 6),
                severity=_classify(psi),
                n_train=sum(1 for v in train_vals if v is not None),
                n_score=sum(1 for v in score_vals if v is not None),
                bins=bins,
            )
        )

    drifts.sort(key=lambda d: d.psi, reverse=True)

    severe = sum(1 for d in drifts if d.severity == "severe")
    moderate = sum(1 for d in drifts if d.severity == "moderate")
    stable = sum(1 for d in drifts if d.severity == "stable")

    if severe > 0:
        verdict: Verdict = "retrain_recommended"
        rationale = (
            f"{severe} feature(s) crossed the severe drift threshold (PSI ≥ "
            f"{_SEVERE}). Retraining the model on a refreshed donor pool is "
            "recommended before serving production predictions."
        )
    elif moderate >= 3:
        verdict = "watch"
        rationale = (
            f"{moderate} feature(s) in the moderate-drift band (0.10 ≤ PSI < "
            f"{_SEVERE}). Monitor; consider a shadow-RCT in the impacted "
            "segments to refresh the donor pool."
        )
    else:
        verdict = "stable"
        rationale = (
            "Scoring distribution is broadly aligned with the training "
            "feature_store. No retraining required."
        )

    max_psi = max((d.psi for d in drifts), default=0.0)
    mean_psi = (
        float(np.mean([d.psi for d in drifts])) if drifts else 0.0
    )

    return {
        "feature_set_version": feature_set_version,
        "n_training_rows": len(training_rows),
        "n_scoring_rows": len(scoring_rows),
        "max_psi": round(max_psi, 6),
        "mean_psi": round(mean_psi, 6),
        "severity_counts": {
            "severe": severe,
            "moderate": moderate,
            "stable": stable,
        },
        "verdict": verdict,
        "rationale": rationale,
        "drifts": [d.to_dict() for d in drifts],
    }
