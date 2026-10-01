"""FastAPI app.

run: uvicorn churn.api:app --app-dir src --port 8000
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException, Query

from . import config
from .predict import get_predictor
from .schemas import CustomerProfile, PredictionResponse

log = logging.getLogger("churn.api")
ModelName = Literal["xgb", "baseline"]


@asynccontextmanager
async def lifespan(app: FastAPI):
    for name in config.MODEL_PATHS:
        get_predictor(name)
    yield


app = FastAPI(title="Telco Churn API", version="1.0.0", lifespan=lifespan)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "models": {name: get_predictor(name).metadata["trained_at"] for name in config.MODEL_PATHS},
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(customer: CustomerProfile, model: ModelName = Query(config.DEFAULT_MODEL)):
    try:
        return get_predictor(model).predict(customer.model_dump())
    except Exception as exc:
        log.exception("prediction failed")
        raise HTTPException(status_code=500, detail="prediction failed") from exc
