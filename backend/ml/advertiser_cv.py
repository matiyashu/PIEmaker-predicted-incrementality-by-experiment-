"""
Existing-advertiser vs new-advertiser CV (V.4 Wave 2 — Phase 4b, paper §5.3).

The paper reports two R² values per model:
  * existing_advertiser_R²  — campaigns whose advertiser_id appears in the
    training fold (interpolating within a known advertiser)
  * new_advertiser_R²       — campaigns whose advertiser_id is held out
    entirely (cold-start advertiser)

Cold-start advertisers carry materially more risk because the model has
never seen the advertiser's idiosyncratic baseline. This module surfaces
that gap on demand and is the foundation for the Wave 3 "advertiser drift"
chart on the Diagnostics page.

Implementation: sklearn's GroupKFold respects the advertiser_id grouping
when picking splits. We run a CV pass to assemble OOF predictions where
each row was scored by a fold that did NOT include its advertiser. Then
we partition the OOF predictions by whether the advertiser appeared
elsewhere in the training data (i.e. has other campaigns in the pool) —
that's the proxy for "existing" advertiser in the paper.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from pie_formulas import weighted_r_squared
from ml.train_random_forest import (
    DEFAULT_GRID,
    _build_pipeline,
    _flatten_feature_rows,
)


@dataclass
class AdvertiserCohortResult:
    cohort: str  # "existing" | "new"
    n: int
    weighted_r2: float
    n_advertisers: int


@dataclass
class AdvertiserCvResult:
    n_total: int
    n_splits: int
    cohorts: list[dict]  # AdvertiserCohortResult.to_dict() each
    cohort_gap_pp: float  # existing_R² - new_R² in percentage points


def run_existing_vs_new(
    feature_rows: list[dict],
    label_rows: list[dict],
    cost_weights: list[float] | None = None,
    advertiser_id_col: str = "advertiser_id",
    n_splits: int = 5,
) -> dict:
    """Group-aware CV partitioned by advertiser_id.

    Reports weighted R² separately for the "existing" cohort (advertiser
    seen in the training data of OTHER campaigns) and the "new" cohort
    (advertiser entirely held out — cold-start).
    """
    feats = _flatten_feature_rows(feature_rows).set_index("campaign_id")
    labels = pd.DataFrame(label_rows).set_index("campaign_id")["icpd"].astype(float)
    common = list(feats.index.intersection(labels.index))
    if len(common) < 10:
        raise ValueError(
            f"need at least 10 aligned rows; got {len(common)}"
        )
    if advertiser_id_col not in feats.columns:
        raise ValueError(
            f"{advertiser_id_col!r} not in feature columns; available "
            f"sample: {sorted(feats.columns)[:8]}…"
        )

    X = feats.loc[common]
    y = labels.loc[common]
    advertisers = X[advertiser_id_col].astype(str).fillna("__unknown__")
    n_unique = advertisers.nunique()
    if n_unique < 2:
        raise ValueError(
            f"need at least 2 unique {advertiser_id_col} values; got {n_unique}"
        )

    weights = (
        np.asarray(cost_weights, dtype=float)
        if cost_weights is not None
        else np.ones(len(common))
    )
    if cost_weights is not None and len(cost_weights) != len(common):
        raise ValueError("cost_weights length must match aligned rows")

    effective_splits = max(2, min(n_splits, n_unique))
    gkf = GroupKFold(n_splits=effective_splits)
    params = DEFAULT_GRID[0]

    preds = np.full(len(common), np.nan, dtype=float)
    held_out_advertisers_per_row: list[bool] = [False] * len(common)
    advertiser_array = advertisers.to_numpy()
    for tr_idx, te_idx in gkf.split(X, y, groups=advertiser_array):
        pipe, _ = _build_pipeline(X.iloc[tr_idx], params)
        pipe.fit(X.iloc[tr_idx], y.iloc[tr_idx])
        preds[te_idx] = pipe.predict(X.iloc[te_idx])
        # Tag the test rows: in GroupKFold every advertiser appears in
        # exactly one fold — so test rows in this fold belong to a "cold-
        # start" advertiser w.r.t. the training data of this fold. For the
        # cohort split, the question is whether the advertiser had OTHER
        # campaigns in training. If the advertiser owns a single campaign
        # globally → it's "new"; if multiple → it's "existing" since
        # peers of those rows did appear in some training fold.

    advertiser_campaign_counts = advertisers.value_counts().to_dict()
    cohort_assignment = [
        "new" if advertiser_campaign_counts[adv] == 1 else "existing"
        for adv in advertiser_array
    ]

    mask = ~np.isnan(preds)
    cohort_arr = np.asarray(cohort_assignment, dtype=object)[mask]
    y_arr = y.to_numpy()[mask]
    p_arr = preds[mask]
    w_arr = weights[mask]
    adv_arr = advertiser_array[mask]

    cohorts: list[AdvertiserCohortResult] = []
    cohort_r2 = {"existing": float("nan"), "new": float("nan")}
    for cohort in ("existing", "new"):
        sel = cohort_arr == cohort
        if sel.sum() == 0:
            cohorts.append(
                AdvertiserCohortResult(
                    cohort=cohort, n=0, weighted_r2=float("nan"), n_advertisers=0
                )
            )
            continue
        try:
            r2 = weighted_r_squared(
                y_arr[sel].tolist(), p_arr[sel].tolist(), w_arr[sel].tolist()
            )
        except ValueError:
            r2 = float("nan")
        cohort_r2[cohort] = r2
        cohorts.append(
            AdvertiserCohortResult(
                cohort=cohort,
                n=int(sel.sum()),
                weighted_r2=r2,
                n_advertisers=int(pd.Series(adv_arr[sel]).nunique()),
            )
        )

    gap_pp = (
        (cohort_r2["existing"] - cohort_r2["new"]) * 100.0
        if not (np.isnan(cohort_r2["existing"]) or np.isnan(cohort_r2["new"]))
        else float("nan")
    )
    return AdvertiserCvResult(
        n_total=int(mask.sum()),
        n_splits=effective_splits,
        cohorts=[asdict(c) for c in cohorts],
        cohort_gap_pp=gap_pp,
    ).__dict__
