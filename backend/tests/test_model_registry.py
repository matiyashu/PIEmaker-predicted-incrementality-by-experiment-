"""Model registry tests (Prompt 2.3)."""

from __future__ import annotations

from pathlib import Path

import pytest

from ml import model_registry
from services.persistence import reset


class _ToyEstimator:
    def __init__(self, k: int):
        self.k = k

    def predict(self, _):
        return [self.k]


@pytest.fixture(autouse=True)
def _reset_state():
    # Clean state files and any persisted .pkl artifacts under state/models/.
    reset()
    for p in model_registry._REGISTRY_DIR.glob("*.pkl"):
        p.unlink()
    yield
    reset()
    for p in model_registry._REGISTRY_DIR.glob("*.pkl"):
        p.unlink()


def test_register_persists_artifact_and_record():
    rec = model_registry.register_model(
        name="pie_rf",
        algorithm="random_forest",
        feature_set_version="v1",
        hyperparameters={"n_estimators": 10},
        training_donor_pool_size=300,
        estimator=_ToyEstimator(7),
    )
    assert rec["status"] == "research"
    assert rec["name"] == "pie_rf"
    assert rec["version_tag"].startswith("v-")
    assert Path(rec["artifact_path"]).exists()


def test_load_model_round_trips_estimator():
    rec = model_registry.register_model(
        name="m",
        algorithm="rf",
        feature_set_version="v1",
        hyperparameters={},
        training_donor_pool_size=0,
        estimator=_ToyEstimator(42),
    )
    loaded = model_registry.load_model(rec["id"])
    assert loaded.predict(None) == [42]


def test_load_model_unknown_raises():
    with pytest.raises(FileNotFoundError):
        model_registry.load_model("nonexistent")


def test_list_models_filters_by_status():
    a = model_registry.register_model(
        name="a", algorithm="rf", feature_set_version="v1", hyperparameters={},
        training_donor_pool_size=0, estimator=_ToyEstimator(1), status="research",
    )
    model_registry.register_model(
        name="b", algorithm="rf", feature_set_version="v1", hyperparameters={},
        training_donor_pool_size=0, estimator=_ToyEstimator(2), status="production",
    )
    research = model_registry.list_models(status="research")
    assert [r["id"] for r in research] == [a["id"]]
    assert len(model_registry.list_models()) == 2


def test_record_metric_and_metrics_for():
    rec = model_registry.register_model(
        name="m", algorithm="rf", feature_set_version="v1", hyperparameters={},
        training_donor_pool_size=0, estimator=_ToyEstimator(1),
    )
    model_registry.record_metric(rec["id"], "weighted_r_squared", 0.81,
                                 ci_lower=0.74, ci_upper=0.87)
    model_registry.record_metric(rec["id"], "ablation_weighted_r2", 0.55,
                                 segment={"spec": "PIE(Pre)"})
    metrics = model_registry.metrics_for(rec["id"])
    assert len(metrics) == 2
    types = {m["metric_type"] for m in metrics}
    assert types == {"weighted_r_squared", "ablation_weighted_r2"}


def test_promote_to_production_flips_status():
    rec = model_registry.register_model(
        name="m", algorithm="rf", feature_set_version="v1", hyperparameters={},
        training_donor_pool_size=0, estimator=_ToyEstimator(1), status="research",
    )
    promoted = model_registry.promote_to_production(rec["id"])
    assert promoted["status"] == "production"
    assert model_registry.list_models(status="production")[0]["id"] == rec["id"]


def test_promote_unknown_raises():
    with pytest.raises(FileNotFoundError):
        model_registry.promote_to_production("ghost")
