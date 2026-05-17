"""
Donor pool size curve (paper Figure 6, V.4 Wave 2 — Phase 3).

For each requested pool size, draw subsamples without replacement, run an
OOF-trained model, and report weighted R² with a bootstrap CI. The
resulting curve answers "how much donor pool do I need before the model
beats the LCC-7D benchmark?" — the central operational question in
paper §5.2.

Implementation notes:
  * For each pool size we draw `n_subsamples` random subsamples (default 5)
    and run a small-k CV per subsample; the headline R² per size is the
    median over subsamples, with p2.5 / p97.5 reported as the CI.
  * `n_splits` defaults to 5 inside this routine — paper's 10-fold curve is
    expensive on small pools and the 5-fold variance is acceptable for
    the shape of the curve.
  * If a requested pool size exceeds the available pool, that point is
    skipped (the curve naturally stops at len(pool)).
"""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from pie_formulas import weighted_r_squared
from ml.train_random_forest import (
    DEFAULT_GRID,
    _build_pipeline,
    _flatten_feature_rows,
)

DEFAULT_SIZES = [50, 100, 200, 400, 800, 1600]


@dataclass
class PoolSizePoint:
    pool_size: int
    weighted_r2_median: float
    weighted_r2_p025: float
    weighted_r2_p975: float
    n_subsamples: int
    n_splits: int


def _oof_r2_for_subsample(
    X: pd.DataFrame,
    y: pd.Series,
    w: np.ndarray,
    n_splits: int,
    seed: int,
) -> float:
    effective_splits = max(2, min(n_splits, len(X) - 1))
    kf = KFold(n_splits=effective_splits, shuffle=True, random_state=seed)
    params = DEFAULT_GRID[0]
    preds = np.full(len(X), np.nan, dtype=float)
    for tr, te in kf.split(X):
        pipe, _ = _build_pipeline(X.iloc[tr], params)
        pipe.fit(X.iloc[tr], y.iloc[tr])
        preds[te] = pipe.predict(X.iloc[te])
    mask = ~np.isnan(preds)
    try:
        return weighted_r_squared(
            y.to_numpy()[mask].tolist(),
            preds[mask].tolist(),
            w[mask].tolist(),
        )
    except ValueError:
        return float("nan")


def run_pool_size_curve(
    feature_rows: list[dict],
    label_rows: list[dict],
    cost_weights: list[float] | None = None,
    sizes: list[int] | None = None,
    n_subsamples: int = 5,
    n_splits: int = 5,
    seed: int = 42,
) -> list[dict]:
    """Return a list of pool-size points for plotting paper Figure 6."""
    feats = _flatten_feature_rows(feature_rows).set_index("campaign_id")
    labels = pd.DataFrame(label_rows).set_index("campaign_id")["icpd"].astype(float)
    common = list(feats.index.intersection(labels.index))
    if len(common) < 5:
        raise ValueError(
            f"need at least 5 aligned (feature, label) rows; got {len(common)}"
        )

    cid_to_weight = (
        {cid: float(w_val) for cid, w_val in zip(common, cost_weights)}
        if cost_weights is not None
        else {cid: 1.0 for cid in common}
    )
    if cost_weights is not None and len(cost_weights) != len(common):
        raise ValueError("cost_weights length must match aligned rows")

    rng = random.Random(seed)
    sizes = sizes or DEFAULT_SIZES
    out: list[PoolSizePoint] = []

    for size in sizes:
        if size > len(common):
            continue
        if size < 5:
            continue
        r2s: list[float] = []
        for s_idx in range(n_subsamples):
            sub = rng.sample(common, size)
            X_sub = feats.loc[sub]
            y_sub = labels.loc[sub]
            w_sub = np.asarray([cid_to_weight[c] for c in sub], dtype=float)
            r2 = _oof_r2_for_subsample(X_sub, y_sub, w_sub, n_splits, seed + s_idx)
            if not np.isnan(r2):
                r2s.append(r2)
        if not r2s:
            continue
        arr = np.asarray(r2s, dtype=float)
        out.append(
            PoolSizePoint(
                pool_size=size,
                weighted_r2_median=float(np.median(arr)),
                weighted_r2_p025=float(np.percentile(arr, 2.5)),
                weighted_r2_p975=float(np.percentile(arr, 97.5)),
                n_subsamples=len(r2s),
                n_splits=max(2, min(n_splits, size - 1)),
            )
        )

    return [asdict(p) for p in out]
