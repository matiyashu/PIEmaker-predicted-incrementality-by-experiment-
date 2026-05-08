"""
Demo seeding (one-click dashboard population).

Runs the full Phase-1→Phase-3 pipeline on the bundled demo CSV so a fresh
backend has populated tables for every workbench page on first load.

Idempotent: each call to seed() resets state, then re-seeds. Used by the
new Dashboard page to bootstrap empty environments (local clones, Vercel
cold starts, etc.).
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from ml import model_registry
from ml.holdout_one_level import SEGMENTATION_VARS, run_extrapolation_test
from services import donor_pool_service as dps
from services.feature_engineering_service import build_features
from services.label_generation_service import generate_labels
from services.model_training_service import train_pie_model
from services.persistence import read_table, reset
from services.prediction_service import score_portfolio
from services.upload_storage import store_upload

DEMO_CSV_PATH = (
    Path(__file__).resolve().parents[2] / "demo" / "piemaker_demo.csv"
)
HOLDOUT_RESULTS_TABLE = "holdout_results"
DEMO_META_TABLE = "demo_meta"


class DemoSeedError(Exception):
    """Raised when the seed pipeline cannot complete (missing CSV, etc.)."""


def _persist_holdout_results(results: list[dict]) -> None:
    from services.persistence import upsert
    now = datetime.now(timezone.utc).isoformat()
    for row in results:
        key_id = f"{row['segmentation_var']}|{row['level']}"
        upsert(
            HOLDOUT_RESULTS_TABLE,
            {**row, "id": key_id, "created_at": now},
            key="id",
        )


def get_status() -> dict:
    meta = read_table(DEMO_META_TABLE)
    seeded_row = meta[0] if meta else None
    return {
        "seeded": seeded_row is not None,
        "last_seeded_at": seeded_row["seeded_at"] if seeded_row else None,
        "model_count": len(read_table("model_versions")),
        "donor_pool_admitted": sum(
            1 for r in read_table("donor_pool_membership") if r.get("admitted")
        ),
        "prediction_run_count": len(read_table("prediction_runs")),
        "holdout_var_count": len(
            {r.get("segmentation_var") for r in read_table(HOLDOUT_RESULTS_TABLE)}
        ),
    }


def _wipe_state() -> None:
    """Clear JSON-shim tables and any persisted model artifacts."""
    reset()
    for p in model_registry._REGISTRY_DIR.glob("*.pkl"):
        p.unlink()


def seed(csv_path: Path | None = None) -> dict:
    """Run the full Phase-1 → Phase-3 pipeline against the demo CSV.

    Returns a summary including durations per stage so the UI can render
    progress.
    """
    csv = csv_path or DEMO_CSV_PATH
    if not csv.exists():
        raise DemoSeedError(
            f"Demo CSV not found at {csv}. Run "
            "`python scripts/generate_demo_csv.py` first."
        )

    durations: dict[str, float] = {}

    _wipe_state()
    durations["wipe"] = 0.0

    # 1. Upload — store + parse so it's accessible via load_upload(upload_id)
    t = time.perf_counter()
    content = csv.read_bytes()
    upload = store_upload(content, csv.name)
    df = pd.read_csv(csv, parse_dates=["start_date", "end_date"])
    durations["upload"] = time.perf_counter() - t

    rct_df = df[df["is_rct"] == 1].copy()
    non_rct_df = df[df["is_rct"] != 1].copy()
    if rct_df.empty:
        raise DemoSeedError("Demo CSV has no RCT rows (is_rct == 1).")

    # 2. Promote RCTs into the donor pool
    t = time.perf_counter()
    promote_rows = rct_df.copy()
    promote_rows["duration_days"] = (
        promote_rows["end_date"] - promote_rows["start_date"]
    ).dt.days
    promote_rows["end_date"] = promote_rows["end_date"].dt.date.astype(str)
    promote_rows["start_date"] = promote_rows["start_date"].dt.date.astype(str)
    for _, row in promote_rows.iterrows():
        rct = row.to_dict()
        dps.promote_rct(rct["campaign_id"], rct)
    durations["donor_pool"] = time.perf_counter() - t

    # 3. Generate labels
    t = time.perf_counter()
    generate_labels(rct_df, has_user_level_data=False)
    durations["labels"] = time.perf_counter() - t

    # 4. Build training features
    t = time.perf_counter()
    build_features(rct_df, mode="training")
    durations["features"] = time.perf_counter() - t

    # 5. Train model (donor pool ≥ 400 → production band, no watermark)
    t = time.perf_counter()
    train_result = train_pie_model(name="pie_random_forest_demo", n_bootstrap=80)
    durations["train"] = time.perf_counter() - t

    # 6. Hold-out-one-level for every segmentation var (so risk badges populate)
    t = time.perf_counter()
    feat_rows = [
        r
        for r in read_table("feature_store")
        if r.get("feature_set_version") == "v1" and r.get("mode") == "training"
    ]
    label_rows = read_table("rct_labels")
    for var in SEGMENTATION_VARS:
        try:
            results = run_extrapolation_test(
                feat_rows, label_rows, var, n_iterations=10
            )
            _persist_holdout_results(results)
        except ValueError:
            # Some segmentation vars may have insufficient level coverage on
            # synthetic data; skip rather than fail the whole seed.
            continue
    durations["holdouts"] = time.perf_counter() - t

    # 7. Portfolio score the non-RCT candidates
    t = time.perf_counter()
    if not non_rct_df.empty:
        non_rct_rows = non_rct_df.copy()
        for col in ("start_date", "end_date"):
            non_rct_rows[col] = non_rct_rows[col].dt.date.astype(str)
        score_portfolio(non_rct_rows.to_dict(orient="records"))
    durations["portfolio"] = time.perf_counter() - t

    # 8. Mark seeded
    from services.persistence import upsert
    upsert(
        DEMO_META_TABLE,
        {
            "id": "singleton",
            "seeded_at": datetime.now(timezone.utc).isoformat(),
            "upload_id": upload.upload_id,
            "model_version_id": train_result["model"]["id"],
        },
        key="id",
    )

    return {
        "upload_id": upload.upload_id,
        "model": train_result["model"],
        "weighted_r_squared": train_result["weighted_r_squared"],
        "n_rcts": int(len(rct_df)),
        "n_non_rcts": int(len(non_rct_df)),
        "donor_pool_band": train_result["donor_pool_status"]["band"],
        "durations": {k: round(v, 3) for k, v in durations.items()},
    }
