"""
Run manifest (V.4 Wave 4 — Phase 7).

Every registered model gets a manifest documenting *what code* produced it
and *which paper alignment* it claims. The manifest is the audit trail
that lets future re-evaluation re-derive the same numbers:

  * ``git_sha``                — commit at training time
  * ``git_dirty``              — True if uncommitted changes were present
  * ``paper_alignment_version``— version stamped on paper_to_code_matrix.json
  * ``python_version``         — interpreter for reproducibility
  * ``hyperparameters``        — passed through verbatim
  * ``model_card_criteria``    — paper-alignment declaration (V.4 Wave 1)
  * ``recorded_at``            — UTC ISO timestamp

This is the smallest manifest that satisfies "could you re-run this in 6
months and get the same numbers?". MLflow artifact tracking + S3 push is
a Wave 4B follow-up; this manifest sits in the model registry alongside
the existing concept_drift_baseline so the trust UI can render it.
"""

from __future__ import annotations

import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


_MATRIX_PATH = Path(__file__).parent / "paper_to_code_matrix.json"
_REPO_ROOT = Path(__file__).resolve().parents[2]


def _git(args: list[str]) -> str:
    try:
        out = subprocess.check_output(
            ["git", *args],
            cwd=str(_REPO_ROOT),
            stderr=subprocess.DEVNULL,
            timeout=5,
        )
        return out.decode("utf-8").strip()
    except (
        subprocess.CalledProcessError,
        subprocess.TimeoutExpired,
        FileNotFoundError,
    ):
        return ""


def _git_sha() -> str:
    return _git(["rev-parse", "HEAD"]) or "unknown"


def _git_dirty() -> bool:
    """Returns True if `git status --porcelain` shows uncommitted changes."""
    status = _git(["status", "--porcelain"])
    return bool(status)


def _paper_alignment_version() -> str:
    """Read the `version` field from paper_to_code_matrix.json."""
    try:
        matrix = json.loads(_MATRIX_PATH.read_text(encoding="utf-8"))
        return matrix.get("version", "unknown")
    except (OSError, json.JSONDecodeError):
        return "unknown"


def build_manifest(
    hyperparameters: dict[str, Any] | None = None,
    model_card_criteria: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a run manifest. Safe to call from any service path; never raises."""
    return {
        "git_sha": _git_sha(),
        "git_dirty": _git_dirty(),
        "paper_alignment_version": _paper_alignment_version(),
        "python_version": platform.python_version(),
        "hyperparameters": hyperparameters or {},
        "model_card_criteria": model_card_criteria or {},
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        **(extra or {}),
    }
