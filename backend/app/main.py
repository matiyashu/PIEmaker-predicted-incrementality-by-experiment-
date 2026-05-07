"""
FastAPI entrypoint.

Phase 0 shipped /health and /info.
Phase 1 wired upload, schema, validate, clean (Prompts 1.1, 1.2).
Phase 2 wired donor-pool, labels, features, models (Prompts 2.1–2.4).
Phase 3 wires predictions (Prompt 3.1).
Later prompts will replace the remaining 501 stubs.
"""

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

import pie_formulas
from app.config import get_settings
from app.routers import clean as clean_router
from app.routers import donor_pool as donor_pool_router
from app.routers import features as features_router
from app.routers import labels as labels_router
from app.routers import models as models_router
from app.routers import decisions as decisions_router
from app.routers import predictions as predictions_router
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

# Phase 2 routers (Prompts 2.1, 2.2, 2.3, 2.4)
app.include_router(donor_pool_router.router)
app.include_router(labels_router.router)
app.include_router(features_router.router)
app.include_router(models_router.router)

# Phase 3 routers (Prompts 3.1, 3.2, 3.3)
app.include_router(predictions_router.router)
app.include_router(decisions_router.router)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "phase": "3", "formulas_version": pie_formulas.__version__}


@app.get("/info")
def info() -> dict:
    return {
        "app": settings.app_name,
        "env": settings.app_env,
        "frozen_formula_count": len(pie_formulas.__all__),
    }


# --- Phase 3+ stubs (PDF §6.3) ---------------------------------------------

_NOT_YET = "Endpoint not yet implemented; arrives in a later prompt."


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
