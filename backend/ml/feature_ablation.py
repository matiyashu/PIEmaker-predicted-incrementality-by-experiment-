"""
Progressive feature ablation matching paper Figure 2 (Prompt 2.3).

Five model specifications:
  PIE(Pre)               — pre-determined features only
  PIE(Pre+Yt)            — adds outcome rate (conversions_per_dollar)
  PIE(Pre+Yt+LCC-7D)     — adds LCC-7D-per-dollar
  PIE(Full)              — all post-determined features
  Raw_LCC_7D_benchmark   — predicts ICPD = LCC-7D / Cost (no model)

Returns weighted R² per spec on a held-out fold so the chart matches the paper.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from pie_formulas import weighted_r_squared
from ml.train_random_forest import (
    DEFAULT_GRID,
    _build_pipeline,
    _flatten_feature_rows,
)
from services.feature_engineering_service import X_PRE_FIELDS

ABLATION_SPECS = [
    "PIE(Pre)",
    "PIE(Pre+Yt)",
    "PIE(Pre+Yt+LCC-7D)",
    "PIE(Full)",
    "Raw_LCC_7D_benchmark",
]


def _columns_for_spec(spec: str) -> list[str] | None:
    """Return the feature columns active in a given spec (None = all)."""
    if spec == "PIE(Pre)":
        return list(X_PRE_FIELDS)
    if spec == "PIE(Pre+Yt)":
        return list(X_PRE_FIELDS) + ["conversions_per_dollar"]
    if spec == "PIE(Pre+Yt+LCC-7D)":
        return list(X_PRE_FIELDS) + ["conversions_per_dollar", "lcc_7d_per_dollar"]
    if spec == "PIE(Full)":
        return None
    if spec == "Raw_LCC_7D_benchmark":
        return ["lcc_7d_per_dollar"]
    raise ValueError(f"unknown spec: {spec}")


@dataclass
class AblationRow:
    spec: str
    weighted_r2: float
    n_train: int
    n_test: int


def run_ablation(
    feature_rows: list[dict],
    label_rows: list[dict],
    cost_weights: list[float] | None = None,
    test_size: float = 0.25,
    seed: int = 42,
) -> list[dict]:
    feats = _flatten_feature_rows(feature_rows).set_index("campaign_id")
    labels = pd.DataFrame(label_rows).set_index("campaign_id")["icpd"].astype(float)
    common = feats.index.intersection(labels.index)
    if len(common) < 4:
        raise ValueError(
            f"need at least 4 aligned (feature, label) rows; got {len(common)}"
        )
    X = feats.loc[common]
    y = labels.loc[common]
    w = (
        np.asarray(cost_weights, dtype=float)
        if cost_weights is not None
        else np.ones(len(common))
    )
    if len(w) != len(common):
        raise ValueError("cost_weights length must match aligned rows")

    Xtr, Xte, ytr, yte, wtr, wte = train_test_split(
        X, y, w, test_size=test_size, random_state=seed
    )

    out: list[AblationRow] = []
    params = DEFAULT_GRID[0]  # cheapest spec for ablation; full grid in train()
    for spec in ABLATION_SPECS:
        if spec == "Raw_LCC_7D_benchmark":
            preds = (
                Xte["lcc_7d_per_dollar"].astype(float).fillna(0.0).to_numpy()
                if "lcc_7d_per_dollar" in Xte.columns
                else np.zeros(len(Xte))
            )
        else:
            cols = _columns_for_spec(spec)
            X_tr_use = Xtr[[c for c in (cols or Xtr.columns) if c in Xtr.columns]]
            X_te_use = Xte[X_tr_use.columns]
            pipe, _ = _build_pipeline(X_tr_use, params)
            pipe.fit(X_tr_use, ytr)
            preds = pipe.predict(X_te_use)
        try:
            r2 = weighted_r_squared(yte.tolist(), preds.tolist(), wte.tolist())
        except ValueError:
            r2 = float("nan")
        out.append(
            AblationRow(spec=spec, weighted_r2=r2, n_train=len(Xtr), n_test=len(Xte))
        )
    return [asdict(r) for r in out]
