"""
Persistence abstraction (V.4 Wave 4 — Phase 7).

Wraps the original JSON-file shim behind a ``PersistenceBackend`` Protocol
so the same callers (`read_table`, `write_table`, `upsert`, `find_by`,
`delete_by`, `reset`) can run against either:

  * **FileShim** — the V.1-V.3 implementation. State lives in
    ``backend/state/<table>.json``. Default. Used by all existing tests.

  * **PostgresBackend** — V.4 production option. State lives in a single
    ``shim_kv`` table keyed by (table_name, key). Activated by setting
    ``PIEMAKER_PERSISTENCE_BACKEND=postgres`` and ``DATABASE_URL``. Keeps
    the shim semantics so existing services don't need to be rewritten —
    individual services can migrate to the typed ORM tables in
    ``backend/alembic/versions/0001_initial_schema.py`` later.

The PostgresBackend gracefully refuses to import its SQLAlchemy
dependencies until it's actually instantiated, so dev runs without a
database installed still work.

Callers should keep using the module-level functions (``read_table``,
``upsert``, etc.) — those delegate to the active backend.
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any, Protocol


# --- Protocol ---------------------------------------------------------------


class PersistenceBackend(Protocol):
    """Stable callback surface for everything that needs persistent state."""

    def read_table(self, name: str) -> list[dict[str, Any]]: ...
    def write_table(self, name: str, rows: list[dict[str, Any]]) -> None: ...
    def upsert(self, name: str, row: dict[str, Any], key: str) -> None: ...
    def find_by(self, name: str, key: str, value: Any) -> dict[str, Any] | None: ...
    def delete_by(self, name: str, key: str, value: Any) -> int: ...
    def reset(self) -> None: ...


# --- FileShim (default, V.1-V.3 behaviour) ----------------------------------


class FileShim:
    """Original JSON-file persistence. One file per table under ``state/``."""

    def __init__(self, state_dir: Path | None = None) -> None:
        self._state_dir = (
            state_dir
            if state_dir is not None
            else Path(__file__).resolve().parents[1] / "state"
        )
        self._state_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def _path(self, name: str) -> Path:
        return self._state_dir / f"{name}.json"

    def read_table(self, name: str) -> list[dict[str, Any]]:
        with self._lock:
            p = self._path(name)
            if not p.exists():
                return []
            try:
                return json.loads(p.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                return []

    def write_table(self, name: str, rows: list[dict[str, Any]]) -> None:
        with self._lock:
            self._path(name).write_text(
                json.dumps(rows, default=str, indent=2), encoding="utf-8"
            )

    def upsert(self, name: str, row: dict[str, Any], key: str) -> None:
        rows = self.read_table(name)
        out = [r for r in rows if r.get(key) != row.get(key)]
        out.append(row)
        self.write_table(name, out)

    def find_by(self, name: str, key: str, value: Any) -> dict[str, Any] | None:
        for r in self.read_table(name):
            if r.get(key) == value:
                return r
        return None

    def delete_by(self, name: str, key: str, value: Any) -> int:
        rows = self.read_table(name)
        out = [r for r in rows if r.get(key) != value]
        n = len(rows) - len(out)
        self.write_table(name, out)
        return n

    def reset(self) -> None:
        """Wipe all state — used in tests."""
        with self._lock:
            for p in self._state_dir.glob("*.json"):
                p.unlink()


# --- PostgresBackend (V.4 production option) --------------------------------


class PostgresBackend:
    """Postgres-backed shim. One JSONB row per (table_name, key) in shim_kv.

    Deliberately *not* using the typed Alembic models from 0001 — those are
    a follow-up. The shim_kv approach gives us a one-line swap from file to
    SQL without touching any service code. Each shim "table" still gets the
    same flexible dict-list semantics.
    """

    def __init__(self, database_url: str) -> None:
        # Defer SQLAlchemy import so the module remains importable when
        # only the FileShim is used.
        try:
            from sqlalchemy import create_engine
            from sqlalchemy.orm import sessionmaker
        except ImportError as exc:  # pragma: no cover — defensive
            raise RuntimeError(
                "PostgresBackend requires sqlalchemy; install it or use "
                "PIEMAKER_PERSISTENCE_BACKEND=file"
            ) from exc

        self._engine = create_engine(database_url, pool_pre_ping=True)
        self._sessionmaker = sessionmaker(bind=self._engine, expire_on_commit=False)

    def _exec(self, sql: str, params: dict | None = None):
        from sqlalchemy import text

        with self._sessionmaker() as session:
            result = session.execute(text(sql), params or {})
            session.commit()
            return result

    def read_table(self, name: str) -> list[dict[str, Any]]:
        rows = self._exec(
            "SELECT row FROM shim_kv WHERE table_name = :n ORDER BY key",
            {"n": name},
        ).fetchall()
        return [r[0] for r in rows]

    def write_table(self, name: str, rows: list[dict[str, Any]]) -> None:
        # Bulk replace: wipe and re-insert. Matches FileShim semantics.
        from sqlalchemy import text

        with self._sessionmaker() as session:
            session.execute(
                text("DELETE FROM shim_kv WHERE table_name = :n"),
                {"n": name},
            )
            for i, row in enumerate(rows):
                # Use a stable index-based key when caller hasn't provided one.
                key = str(row.get("id") or row.get("campaign_id") or i)
                session.execute(
                    text(
                        "INSERT INTO shim_kv (table_name, key, row) "
                        "VALUES (:n, :k, CAST(:r AS JSONB))"
                    ),
                    {"n": name, "k": key, "r": json.dumps(row, default=str)},
                )
            session.commit()

    def upsert(self, name: str, row: dict[str, Any], key: str) -> None:
        key_value = row.get(key)
        if key_value is None:
            raise ValueError(f"upsert row missing key column {key!r}")
        from sqlalchemy import text

        with self._sessionmaker() as session:
            session.execute(
                text(
                    "INSERT INTO shim_kv (table_name, key, row) "
                    "VALUES (:n, :k, CAST(:r AS JSONB)) "
                    "ON CONFLICT (table_name, key) DO UPDATE "
                    "SET row = EXCLUDED.row"
                ),
                {
                    "n": name,
                    "k": str(key_value),
                    "r": json.dumps(row, default=str),
                },
            )
            session.commit()

    def find_by(self, name: str, key: str, value: Any) -> dict[str, Any] | None:
        from sqlalchemy import text

        with self._sessionmaker() as session:
            row = session.execute(
                text(
                    "SELECT row FROM shim_kv WHERE table_name = :n "
                    "AND row ->> :k = :v LIMIT 1"
                ),
                {"n": name, "k": key, "v": str(value)},
            ).fetchone()
            return row[0] if row else None

    def delete_by(self, name: str, key: str, value: Any) -> int:
        from sqlalchemy import text

        with self._sessionmaker() as session:
            result = session.execute(
                text(
                    "DELETE FROM shim_kv WHERE table_name = :n "
                    "AND row ->> :k = :v"
                ),
                {"n": name, "k": key, "v": str(value)},
            )
            session.commit()
            return int(result.rowcount or 0)

    def reset(self) -> None:
        from sqlalchemy import text

        with self._sessionmaker() as session:
            session.execute(text("DELETE FROM shim_kv"))
            session.commit()


# --- Backend resolution -----------------------------------------------------


def _resolve_backend() -> PersistenceBackend:
    """Pick the active backend at import time, based on env vars.

    Default: ``file``. The Postgres path requires both
    ``PIEMAKER_PERSISTENCE_BACKEND=postgres`` AND ``DATABASE_URL`` to be set;
    if either is missing we fall back to ``file`` and emit a one-shot
    warning so the caller sees the deviation in logs.
    """
    choice = os.getenv("PIEMAKER_PERSISTENCE_BACKEND", "file").lower()
    if choice == "postgres":
        url = os.getenv("DATABASE_URL")
        if not url:
            import warnings

            warnings.warn(
                "PIEMAKER_PERSISTENCE_BACKEND=postgres but DATABASE_URL is "
                "unset; falling back to file shim.",
                RuntimeWarning,
                stacklevel=2,
            )
            return FileShim()
        return PostgresBackend(url)
    return FileShim()


_BACKEND: PersistenceBackend = _resolve_backend()


def get_backend() -> PersistenceBackend:
    """Return the currently active backend (for tests / introspection)."""
    return _BACKEND


def set_backend(backend: PersistenceBackend) -> None:
    """Override the active backend. Used by tests to inject a clean FileShim."""
    global _BACKEND
    _BACKEND = backend


# --- Public API (preserved verbatim from V.1) -------------------------------


def read_table(name: str) -> list[dict[str, Any]]:
    return _BACKEND.read_table(name)


def write_table(name: str, rows: list[dict[str, Any]]) -> None:
    _BACKEND.write_table(name, rows)


def upsert(name: str, row: dict[str, Any], key: str) -> None:
    _BACKEND.upsert(name, row, key)


def find_by(name: str, key: str, value: Any) -> dict[str, Any] | None:
    return _BACKEND.find_by(name, key, value)


def delete_by(name: str, key: str, value: Any) -> int:
    return _BACKEND.delete_by(name, key, value)


def reset() -> None:
    """Wipe all state — used in tests."""
    _BACKEND.reset()
