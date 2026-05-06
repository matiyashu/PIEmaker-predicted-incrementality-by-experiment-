"""
FastAPI entrypoint.

Phase 0 shipped /health and /info.
Phase 1 wires the upload, schema, validate, and clean routers (Prompts 1.1, 1.2).
Later prompts will replace the remaining 501 stubs.
"""

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

import pie_formulas
from app.config import get_settings
from app.routers import clean as clean_router
from app.routers import schema_map as schema_router
from app.routers import upload as upload_router
from app.routers import validate as validate_router

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

# Phase 1 routers (Prompts 1.1, 1.2)
app.include_router(upload_router.router)
app.include_router(schema_router.router)
app.include_router(validate_router.router)
app.include_router(clean_router.router)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "phase": "1", "formulas_version": pie_formulas.__version__}


@app.get("/info")
def info() -> dict:
    return {
        "app": settings.app_name,
        "env": settings.app_env,
        "frozen_formula_count": len(pie_formulas.__all__),
    }


# --- Phase 2+ stubs (PDF §6.3) ---------------------------------------------

_NOT_YET = "Endpoint not yet implemented; arrives in a later prompt."


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
