"""Demo-seed tests."""

from __future__ import annotations

import pytest

from ml import model_registry
from services import demo_seed_service as ds
from services.persistence import read_table, reset


@pytest.fixture(autouse=True)
def _reset_state():
    reset()
    for p in model_registry._REGISTRY_DIR.glob("*.pkl"):
        p.unlink()
    yield
    reset()
    for p in model_registry._REGISTRY_DIR.glob("*.pkl"):
        p.unlink()


def test_status_reports_unseeded_when_empty():
    s = ds.get_status()
    assert s["seeded"] is False
    assert s["model_count"] == 0
    assert s["donor_pool_admitted"] == 0
    assert s["prediction_run_count"] == 0


def test_seed_populates_every_table():
    """End-to-end seed produces a populated state across all key tables."""
    result = ds.seed()
    assert result["n_rcts"] >= 200  # demo CSV has 400
    assert result["n_non_rcts"] >= 1
    assert result["donor_pool_band"] in ("production", "production_full")
    assert "model" in result
    assert "weighted_r_squared" in result
    assert "durations" in result

    s = ds.get_status()
    assert s["seeded"] is True
    assert s["last_seeded_at"] is not None
    assert s["model_count"] >= 1
    assert s["donor_pool_admitted"] >= 200
    assert s["prediction_run_count"] >= 1
    assert s["holdout_var_count"] >= 1

    # Spot-check a few tables
    assert read_table("rct_labels"), "rct_labels should have rows"
    assert read_table("feature_store"), "feature_store should have rows"
    assert read_table("model_versions"), "model_versions should have rows"
    assert read_table("prediction_runs"), "prediction_runs should have rows"
    assert read_table("holdout_results"), "holdout_results should have rows"


def test_seed_is_idempotent():
    """Calling seed() twice should fully replace state, not duplicate."""
    first = ds.seed()
    n_models_after_first = len(read_table("model_versions"))
    second = ds.seed()
    n_models_after_second = len(read_table("model_versions"))
    assert n_models_after_second == n_models_after_first
    # Different model id (new UUID) confirms a fresh registration, not a duplicate.
    assert first["model"]["id"] != second["model"]["id"]
