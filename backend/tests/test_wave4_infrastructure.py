"""V.4 Wave 4 — production hardening tests.

Covers:
  * run_manifest captures git SHA + paper-alignment version + python version
  * persistence Protocol: FileShim works through public API
  * persistence Protocol: set_backend swaps the active backend
  * PostgresBackend module imports without sqlalchemy installed
    (deferred import — instantiation, not module import, is what requires it)
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pie_formulas.run_manifest import build_manifest
from services import persistence


def test_run_manifest_contains_required_keys():
    manifest = build_manifest()
    for key in (
        "git_sha",
        "git_dirty",
        "paper_alignment_version",
        "python_version",
        "hyperparameters",
        "model_card_criteria",
        "recorded_at",
    ):
        assert key in manifest, f"missing manifest key: {key}"


def test_run_manifest_paper_alignment_version_is_v4():
    """paper_to_code_matrix.json declares version v4.0.0 since Wave 1."""
    manifest = build_manifest()
    # Either the actual version or "unknown" if the file's missing in CI.
    assert manifest["paper_alignment_version"] in {"v4.0.0", "unknown"}


def test_run_manifest_passes_through_hyperparameters():
    hp = {"n_estimators": 200, "max_depth": 10}
    manifest = build_manifest(hyperparameters=hp)
    assert manifest["hyperparameters"] == hp


def test_run_manifest_extra_merges():
    manifest = build_manifest(extra={"custom_field": 42})
    assert manifest["custom_field"] == 42


def test_run_manifest_never_raises_outside_repo(tmp_path, monkeypatch):
    """If git isn't available, manifest still returns with `unknown`."""
    # Simulate git missing by pointing PATH at empty tmp_path.
    monkeypatch.setenv("PATH", str(tmp_path))
    manifest = build_manifest()
    # git_sha should fall back gracefully — either "unknown" or whatever git
    # returned from the repo root before we clobbered PATH (subprocess uses
    # absolute path resolution so behavior depends on system).
    assert isinstance(manifest["git_sha"], str)


# --- persistence Protocol -----------------------------------------------------


def test_default_backend_is_file_shim():
    assert isinstance(persistence.get_backend(), persistence.FileShim)


def test_file_shim_round_trip(tmp_path):
    shim = persistence.FileShim(state_dir=tmp_path)
    shim.upsert("things", {"id": "a", "v": 1}, key="id")
    shim.upsert("things", {"id": "b", "v": 2}, key="id")
    shim.upsert("things", {"id": "a", "v": 99}, key="id")  # update
    rows = shim.read_table("things")
    assert len(rows) == 2
    by_id = {r["id"]: r["v"] for r in rows}
    assert by_id == {"a": 99, "b": 2}
    assert shim.find_by("things", "id", "b") == {"id": "b", "v": 2}
    n = shim.delete_by("things", "id", "a")
    assert n == 1
    shim.reset()
    assert shim.read_table("things") == []


def test_set_backend_swaps_active_implementation(tmp_path):
    """Public-API functions delegate to whatever set_backend() last set."""
    custom = persistence.FileShim(state_dir=tmp_path)
    original = persistence.get_backend()
    try:
        persistence.set_backend(custom)
        persistence.upsert("foo", {"id": "x", "n": 7}, key="id")
        assert persistence.read_table("foo") == [{"id": "x", "n": 7}]
    finally:
        persistence.set_backend(original)


def test_postgres_backend_module_imports_without_sqlalchemy():
    """The PersistenceBackend module must be importable even when
    SQLAlchemy isn't installed — sqlalchemy imports live inside
    PostgresBackend.__init__ and per-method bodies."""
    from services.persistence import PostgresBackend  # noqa: F401
    # Importing the class should not raise. Instantiating it requires
    # SQLAlchemy + a valid URL — that's deferred.


def test_postgres_resolution_warns_when_url_missing(monkeypatch):
    """PIEMAKER_PERSISTENCE_BACKEND=postgres without DATABASE_URL falls back
    to FileShim with a warning rather than crashing."""
    monkeypatch.setenv("PIEMAKER_PERSISTENCE_BACKEND", "postgres")
    monkeypatch.delenv("DATABASE_URL", raising=False)
    with pytest.warns(RuntimeWarning):
        backend = persistence._resolve_backend()
    assert isinstance(backend, persistence.FileShim)
