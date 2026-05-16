"""
Random Forest training (PDF Eq. 20, p. 16; Prompt 2.3 + V.4 Wave 1).

V.4: paper-mode evaluation requires headline R² computed from **out-of-fold**
predictions, not a full-data refit. `train()` now assembles per-fold
predictions across the CV loop into `TrainingArtifacts.oof_predictions` so
the downstream orchestrator can compute weighted R² on those instead of
in-sample. Default `n_splits=10` matches paper §5.1.

The trade-off: with multiple hyperparameter specs in the grid, we keep the
OOF predictions from whichever spec scored best on CV — no extra training
passes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from services.feature_engineering_service import X_POST_FIELDS, X_PRE_FIELDS

# Grid kept small so Phase 2.3 trains quickly on small donor pools; widen
# (and switch to Optuna) once the pool exceeds ~400 RCTs.
DEFAULT_GRID = [
    {"n_estimators": 200, "max_depth": 10, "min_samples_leaf": 3, "max_features": "sqrt"},
    {"n_estimators": 400, "max_depth": 12, "min_samples_leaf": 5, "max_features": 0.5},
    {"n_estimators": 800, "max_depth": 16, "min_samples_leaf": 1, "max_features": "sqrt"},
]


@dataclass
class TrainingArtifacts:
    estimator: Pipeline
    feature_columns: list[str]
    cv_scores: list[float]
    chosen_hyperparameters: dict
    n_observations: int
    # V.4 Wave 1: OOF predictions assembled during CV. The orchestrator
    # uses these (not a full-data refit) to compute the paper-aligned
    # headline R². `oof_campaign_ids` lets callers re-align with labels.
    oof_predictions: list[float] | None = None
    oof_campaign_ids: list[str] | None = None
    oof_n_splits: int | None = None


def _flatten_feature_rows(feature_rows: list[dict]) -> pd.DataFrame:
    """Turn feature_store rows ({campaign_id, x_pre, x_post, ...}) into a flat frame."""
    flat = []
    for r in feature_rows:
        rec = {"campaign_id": r["campaign_id"]}
        rec.update(r.get("x_pre") or {})
        rec.update(r.get("x_post") or {})
        flat.append(rec)
    return pd.DataFrame(flat)


def _build_pipeline(
    df: pd.DataFrame, hyperparameters: dict
) -> tuple[Pipeline, list[str]]:
    available = [c for c in X_PRE_FIELDS + X_POST_FIELDS if c in df.columns]
    cat_cols = [c for c in available if df[c].dtype == "object" or pd.api.types.is_string_dtype(df[c])]
    num_cols = [c for c in available if c not in cat_cols]

    transformer = ColumnTransformer(
        [
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler(with_mean=True, with_std=True)),
                    ]
                ),
                num_cols,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                cat_cols,
            ),
        ],
        remainder="drop",
    )
    rf = RandomForestRegressor(
        n_estimators=hyperparameters["n_estimators"],
        max_depth=hyperparameters["max_depth"],
        min_samples_leaf=hyperparameters["min_samples_leaf"],
        max_features=hyperparameters["max_features"],
        n_jobs=-1,
        random_state=42,
    )
    pipe = Pipeline([("preprocess", transformer), ("rf", rf)])
    return pipe, available


def train(
    feature_rows: list[dict],
    label_rows: list[dict],
    weights: list[float] | None = None,
    grid: list[dict] | None = None,
    n_splits: int = 10,
    return_oof: bool = True,
) -> TrainingArtifacts:
    """Cross-validated grid search; refit on full data with the best params.

    V.4: when ``return_oof`` is True (the default), per-fold predictions for
    the best-scoring hyperparameter set are assembled into
    ``TrainingArtifacts.oof_predictions``. The orchestrator should use those
    for the paper-aligned headline R² and bootstrap CI — never the full-data
    refit (which is in-sample by construction).

    The default ``n_splits=10`` matches paper §5.1. For small pools the
    effective split count is bounded by ``len(common) - 1``.
    """
    if not feature_rows or not label_rows:
        raise ValueError("feature_rows and label_rows must be non-empty")

    feats = _flatten_feature_rows(feature_rows)
    labels = pd.DataFrame(label_rows).set_index("campaign_id")["icpd"]
    feats = feats.set_index("campaign_id")
    common = feats.index.intersection(labels.index)
    if len(common) < 5:
        raise ValueError(
            f"need at least 5 aligned (feature, label) rows; got {len(common)}"
        )
    X = feats.loc[common]
    y = labels.loc[common].astype(float)
    w = (
        np.asarray(weights, dtype=float)
        if weights is not None
        else np.ones(len(common))
    )

    grid = grid or DEFAULT_GRID
    effective_splits = min(n_splits, max(2, len(common) - 1))
    kf = KFold(n_splits=effective_splits, shuffle=True, random_state=42)

    best_params = grid[0]
    best_mean = -np.inf
    best_scores: list[float] = []
    # V.4: collect OOF predictions for each candidate; keep the winner's.
    best_oof = np.full(len(common), np.nan, dtype=float)
    for params in grid:
        scores: list[float] = []
        oof_preds = np.full(len(common), np.nan, dtype=float)
        for tr_idx, te_idx in kf.split(X):
            pipe, _ = _build_pipeline(X.iloc[tr_idx], params)
            pipe.fit(X.iloc[tr_idx], y.iloc[tr_idx])
            preds = pipe.predict(X.iloc[te_idx])
            oof_preds[te_idx] = preds
            scores.append(pipe.score(X.iloc[te_idx], y.iloc[te_idx]))
        mean = float(np.mean(scores))
        if mean > best_mean:
            best_mean = mean
            best_params = params
            best_scores = scores
            best_oof = oof_preds

    final_pipe, columns = _build_pipeline(X, best_params)
    final_pipe.fit(X, y, **({"rf__sample_weight": w} if hasattr(final_pipe, "rf") else {}))
    return TrainingArtifacts(
        estimator=final_pipe,
        feature_columns=columns,
        cv_scores=best_scores,
        chosen_hyperparameters=best_params,
        n_observations=len(common),
        oof_predictions=best_oof.tolist() if return_oof else None,
        oof_campaign_ids=[str(cid) for cid in common] if return_oof else None,
        oof_n_splits=effective_splits if return_oof else None,
    )


def predict_icpd(
    estimator: Pipeline, feature_rows: list[dict]
) -> np.ndarray:
    feats = _flatten_feature_rows(feature_rows)
    if "campaign_id" in feats.columns:
        feats = feats.set_index("campaign_id")
    return estimator.predict(feats)
