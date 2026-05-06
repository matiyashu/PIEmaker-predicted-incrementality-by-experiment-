"""
JSON-file persistence shim for Phase 2.

Acts as a stand-in for the SQLAlchemy/Postgres tables defined in the Alembic
migration. Same conceptual shape, swap-in-able once Docker + Postgres are
wired up: every function maps 1:1 to a future SQLAlchemy CRUD call.

Tables (file-per-table):
  rct_labels                        Phase 2.2
  feature_store                     Phase 2.2
  model_versions                    Phase 2.3
  model_metrics                     Phase 2.3
  prediction_runs                   Phase 3.1 (placeholder)
  shadow_rct_recommendations        Phase 2.1
  donor_pool_membership             Phase 2.1
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

_STATE_DIR = Path(__file__).resolve().parents[1] / "state"
_STATE_DIR.mkdir(parents=True, exist_ok=True)

_LOCK = threading.Lock()


def _path(name: str) -> Path:
    return _STATE_DIR / f"{name}.json"


def read_table(name: str) -> list[dict[str, Any]]:
    with _LOCK:
        p = _path(name)
        if not p.exists():
            return []
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return []


def write_table(name: str, rows: list[dict[str, Any]]) -> None:
    with _LOCK:
        _path(name).write_text(json.dumps(rows, default=str, indent=2), encoding="utf-8")


def upsert(name: str, row: dict[str, Any], key: str) -> None:
    rows = read_table(name)
    out = [r for r in rows if r.get(key) != row.get(key)]
    out.append(row)
    write_table(name, out)


def find_by(name: str, key: str, value: Any) -> dict[str, Any] | None:
    for r in read_table(name):
        if r.get(key) == value:
            return r
    return None


def delete_by(name: str, key: str, value: Any) -> int:
    rows = read_table(name)
    out = [r for r in rows if r.get(key) != value]
    n = len(rows) - len(out)
    write_table(name, out)
    return n


def reset() -> None:
    """Wipe all state — used in tests."""
    with _LOCK:
        for p in _STATE_DIR.glob("*.json"):
            p.unlink()
