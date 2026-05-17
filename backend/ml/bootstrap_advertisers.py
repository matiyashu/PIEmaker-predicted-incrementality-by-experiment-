"""
Bootstrap over advertisers (V.4 Wave 2 — Phase 4c, paper §5.3).

Paper-faithful uncertainty estimate. The V.3 bootstrap resamples *rows*
(campaigns) which understates uncertainty when multiple campaigns share an
advertiser — they're correlated. The paper-correct version resamples
*advertisers* (clusters) with replacement: for each draw, sample a
multiset of advertiser_ids, build the training set from all campaigns
those advertisers own, evaluate weighted R² on the OOF predictions for
the held-out advertisers.

This gives wider, more honest CIs than the V.3 row bootstrap, especially
when the donor pool is dominated by a small number of large advertisers.
"""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from pie_formulas import weighted_r_squared
from ml.train_random_forest import (
    DEFAULT_GRID,
    _build_pipeline,
    _flatten_feature_rows,
)


@dataclass
class AdvertiserBootstrapResult:
    n_draws: int
    n_advertisers: int
    mean: float
    p025: float
    p975: float
    distribution: list[float]


def bootstrap_over_advertisers(
    feature_rows: list[dict],
    label_rows: list[dict],
    cost_weights: list[float] | None = None,
    advertiser_id_col: str = "advertiser_id",
    n_draws: int = 100,
    seed: int = 42,
) -> dict:
    """Cluster-bootstrap CI for the weighted R² of the RF model.

    For each draw: sample advertisers with replacement, build the training
    set from their campaigns, predict on the campaigns of advertisers NOT
    drawn, compute weighted R². Repeat `n_draws` times.
    """
    feats = _flatten_feature_rows(feature_rows).set_index("campaign_id")
    labels = pd.DataFrame(label_rows).set_index("campaign_id")["icpd"].astype(float)
    common = list(feats.index.intersection(labels.index))
    if len(common) < 10:
        raise ValueError(
            f"need at least 10 aligned rows; got {len(common)}"
        )
    if advertiser_id_col not in feats.columns:
        raise ValueError(f"{advertiser_id_col!r} not in feature columns")

    X = feats.loc[common]
    y = labels.loc[common]
    advertisers_full = X[advertiser_id_col].astype(str).fillna("__unknown__").to_numpy()
    unique_advertisers = pd.unique(advertisers_full).tolist()
    n_unique = len(unique_advertisers)
    if n_unique < 3:
        raise ValueError(
            f"need at least 3 unique {advertiser_id_col} values; got {n_unique}"
        )

    weights = (
        np.asarray(cost_weights, dtype=float)
        if cost_weights is not None
        else np.ones(len(common))
    )
    if cost_weights is not None and len(cost_weights) != len(common):
        raise ValueError("cost_weights length must match aligned rows")

    params = DEFAULT_GRID[0]
    rng = random.Random(seed)
    distribution: list[float] = []

    # Pre-compute advertiser → row indices
    adv_to_rows: dict[str, list[int]] = {}
    for i, adv in enumerate(advertisers_full):
        adv_to_rows.setdefault(adv, []).append(i)

    for _ in range(n_draws):
        sampled = [rng.choice(unique_advertisers) for _ in range(n_unique)]
        sampled_set = set(sampled)
        train_idx = [
            idx
            for adv in sampled
            for idx in adv_to_rows.get(adv, [])
        ]
        test_idx = [
            idx
            for adv in unique_advertisers
            if adv not in sampled_set
            for idx in adv_to_rows.get(adv, [])
        ]
        if not train_idx or not test_idx:
            continue
        try:
            pipe, _ = _build_pipeline(X.iloc[train_idx], params)
            pipe.fit(X.iloc[train_idx], y.iloc[train_idx])
            preds = pipe.predict(X.iloc[test_idx])
            r2 = weighted_r_squared(
                y.iloc[test_idx].tolist(),
                preds.tolist(),
                weights[test_idx].tolist(),
            )
        except (ValueError, RuntimeError):
            continue
        if not np.isnan(r2):
            distribution.append(float(r2))

    if not distribution:
        raise ValueError(
            "no valid bootstrap draws produced; pool may be too small or "
            "advertiser_id may be too uniform"
        )

    arr = np.asarray(distribution, dtype=float)
    return asdict(
        AdvertiserBootstrapResult(
            n_draws=len(distribution),
            n_advertisers=n_unique,
            mean=float(np.mean(arr)),
            p025=float(np.percentile(arr, 2.5)),
            p975=float(np.percentile(arr, 97.5)),
            distribution=distribution,
        )
    )
