"""
FastAPI entrypoint — Phase 0 stub.

Phase 0 ships /health and /info; all other endpoints from PDF §6.3 return
501 Not Implemented and are filled in by Prompts 1.x onward. This file proves
the formula library imports cleanly inside the API process.
"""

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

import pie_formulas
from app.config import get_settings

settings = get_settings()

app = FastAPI(
    title=settings.app_name,
    version=pie_formulas.__version__,
    description=(
        "PIE Measurement Workbench — campaign-level incrementality prediction "
        "platform built on Gordon, Moakler & Zettelmeyer (NBER w35044, 2026)."
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "phase": "0", "formulas_version": pie_formulas.__version__}


@app.get("/info")
def info() -> dict:
    return {
        "app": settings.app_name,
        "env": settings.app_env,
        "frozen_formula_count": len(pie_formulas.__all__),
    }


# --- Phase 1+ stubs (PDF §6.3) ---------------------------------------------

_NOT_YET = "Endpoint not yet implemented in Phase 0; arrives in a later prompt."


@app.post("/api/upload")
def upload() -> None:
    raise HTTPException(status_code=501, detail=_NOT_YET)


@app.post("/api/schema/map")
def schema_map() -> None:
    raise HTTPException(status_code=501, detail=_NOT_YET)


@app.post("/api/validate")
def validate() -> None:
    raise HTTPException(status_code=501, detail=_NOT_YET)


@app.post("/api/clean")
def clean() -> None:
    raise HTTPException(status_code=501, detail=_NOT_YET)


@app.get("/api/donor-pool")
def donor_pool() -> None:
    raise HTTPException(status_code=501, detail=_NOT_YET)


@app.post("/api/labels/generate")
def labels_generate() -> None:
    raise HTTPException(status_code=501, detail=_NOT_YET)


@app.post("/api/features/build")
def features_build() -> None:
    raise HTTPException(status_code=501, detail=_NOT_YET)


@app.post("/api/models/train")
def models_train() -> None:
    raise HTTPException(status_code=501, detail=_NOT_YET)


@app.post("/api/predict")
def predict() -> None:
    raise HTTPException(status_code=501, detail=_NOT_YET)


@app.post("/api/decision/simulate")
def decision_simulate() -> None:
    raise HTTPException(status_code=501, detail=_NOT_YET)


@app.post("/api/decision/expected-cost")
def decision_expected_cost() -> None:
    raise HTTPException(status_code=501, detail=_NOT_YET)


@app.post("/api/monitoring/drift")
def monitoring_drift() -> None:
    raise HTTPException(status_code=501, detail=_NOT_YET)
