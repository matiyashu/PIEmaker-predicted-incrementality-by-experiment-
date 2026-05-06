"""
File-based model registry (Prompt 2.3).

Stand-in for MLflow + Postgres. Persists model artifacts and metadata under
backend/state/models/. Same conceptual API; an MLflow-backed swap-in arrives
once the Phase-1 docker-compose Postgres is up.
"""

from __future__ import annotations

import pickle
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from services.persistence import read_table, upsert

_REGISTRY_DIR = Path(__file__).resolve().parents[1] / "state" / "models"
_REGISTRY_DIR.mkdir(parents=True, exist_ok=True)
MODEL_VERSIONS_TABLE = "model_versions"
MODEL_METRICS_TABLE = "model_metrics"


def _next_version_id() -> str:
    return uuid.uuid4().hex[:12]


def register_model(
    name: str,
    algorithm: str,
    feature_set_version: str,
    hyperparameters: dict,
    training_donor_pool_size: int,
    estimator: Any,
    concept_drift_baseline: dict | None = None,
    status: str = "research",
) -> dict:
    version_id = _next_version_id()
    artifact_path = _REGISTRY_DIR / f"{version_id}.pkl"
    with open(artifact_path, "wb") as f:
        pickle.dump(estimator, f)
    record = {
        "id": version_id,
        "name": name,
        "version_tag": f"v-{version_id[:6]}",
        "status": status,
        "algorithm": algorithm,
        "feature_set_version": feature_set_version,
        "hyperparameters": hyperparameters,
        "training_donor_pool_size": training_donor_pool_size,
        "concept_drift_baseline": concept_drift_baseline,
        "artifact_path": str(artifact_path),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    upsert(MODEL_VERSIONS_TABLE, record, key="id")
    return record


def load_model(model_version_id: str) -> Any:
    record = next(
        (r for r in read_table(MODEL_VERSIONS_TABLE) if r["id"] == model_version_id),
        None,
    )
    if record is None:
        raise FileNotFoundError(f"model version {model_version_id} not found")
    with open(record["artifact_path"], "rb") as f:
        return pickle.load(f)


def list_models(status: str | None = None) -> list[dict]:
    rows = read_table(MODEL_VERSIONS_TABLE)
    if status:
        rows = [r for r in rows if r.get("status") == status]
    return rows


def record_metric(
    model_version_id: str,
    metric_type: str,
    value: float,
    ci_lower: float | None = None,
    ci_upper: float | None = None,
    segment: dict | None = None,
) -> dict:
    record = {
        "id": uuid.uuid4().hex[:12],
        "model_version_id": model_version_id,
        "metric_type": metric_type,
        "value": value,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "segment": segment,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    upsert(MODEL_METRICS_TABLE, record, key="id")
    return record


def metrics_for(model_version_id: str) -> list[dict]:
    return [
        r
        for r in read_table(MODEL_METRICS_TABLE)
        if r["model_version_id"] == model_version_id
    ]


def promote_to_production(model_version_id: str) -> dict:
    record = next(
        (r for r in read_table(MODEL_VERSIONS_TABLE) if r["id"] == model_version_id),
        None,
    )
    if record is None:
        raise FileNotFoundError(f"model version {model_version_id} not found")
    record["status"] = "production"
    upsert(MODEL_VERSIONS_TABLE, record, key="id")
    return record
