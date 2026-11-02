"""FastAPI inference service for the monthly price forecast.

Run:   uvicorn main:app --reload        (from deploy/fastapi_service)
Docs:  http://127.0.0.1:8000/docs
"""
from __future__ import annotations

import json
import os
import pickle
import threading
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

ARTIFACT_DIR = Path(os.getenv("ARTIFACT_DIR", Path(__file__).resolve().parents[2] / "artifacts"))
RELOAD_TOKEN = os.getenv("RELOAD_TOKEN", "change-me")
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000").split(",")
Z80, Z95 = 1.2816, 1.96


class _State:
    model = None
    meta: dict = {}
    lock = threading.Lock()


state = _State()


def load_artifact() -> None:
    with open(ARTIFACT_DIR / "arima_110_log.pkl", "rb") as f:
        model = pickle.load(f)
    meta = json.loads((ARTIFACT_DIR / "model_meta.json").read_text())
    with state.lock:  # atomic hot swap
        state.model, state.meta = model, meta


# ---------------------------------------------------------------- schemas ---
class MonthlyPrice(BaseModel):
    month: date = Field(..., description="Any day in the month, e.g. 2025-10-01")
    price: float = Field(..., gt=0)

    @field_validator("month")
    @classmethod
    def first_of_month(cls, v: date) -> date:
        return v.replace(day=1)


class PredictRequest(BaseModel):
    horizon: int = Field(12, ge=1, le=24, description="Months to forecast")
    recent_prices: list[MonthlyPrice] = Field(
        default_factory=list,
        description="Optional prices for months after the model's training window, oldest first. "
                    "They update the forecast origin in real time without refitting.")


class ForecastPoint(BaseModel):
    month: str
    mean: float
    lo80: float
    hi80: float
    lo95: float
    hi95: float


class PredictResponse(BaseModel):
    model_version: str
    origin_month: str
    horizon: int
    forecast: list[ForecastPoint]


# -------------------------------------------------------------------- app ---
app = FastAPI(title="Way2Agri Price Forecast API", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=ALLOWED_ORIGINS, allow_methods=["GET", "POST"],
                   allow_headers=["*"])


@app.on_event("startup")
def _startup() -> None:
    load_artifact()


def _last_month(model) -> pd.Timestamp:
    return pd.Timestamp(model.model._index[-1])


@app.get("/health")
def health():
    return {"status": "ok", "model_version": state.meta.get("version"),
            "trained_through": state.meta.get("trained_through")}


@app.get("/model")
def model_info():
    keys = ("model", "version", "trained_through", "n_obs", "phi", "sigma", "holdout_mape_pct", "cv_mape_pct")
    return {k: state.meta.get(k) for k in keys}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    with state.lock:
        model, meta = state.model, state.meta
    if model is None:
        raise HTTPException(503, "Model not loaded")

    last = _last_month(model)
    if req.recent_prices:
        months = [pd.Timestamp(p.month) for p in req.recent_prices]
        expected = pd.date_range(last + pd.offsets.MonthBegin(1), periods=len(months), freq="MS")
        if list(expected) != months:
            raise HTTPException(
                422, f"recent_prices must be consecutive months starting {expected[0]:%Y-%m}")
        new = pd.Series(np.log([p.price for p in req.recent_prices]), index=expected,
                        name=model.model.endog_names)
        model = model.append(new, refit=False)  # update state, keep trained parameters
        last = expected[-1]

    f = model.get_forecast(req.horizon)
    mean = np.asarray(f.predicted_mean)
    sd = np.sqrt(np.asarray(f.var_pred_mean))
    idx = pd.date_range(last + pd.offsets.MonthBegin(1), periods=req.horizon, freq="MS")
    points = [ForecastPoint(month=f"{d:%Y-%m}", mean=round(float(np.exp(m)), 1),
                            lo80=round(float(np.exp(m - Z80 * s)), 1), hi80=round(float(np.exp(m + Z80 * s)), 1),
                            lo95=round(float(np.exp(m - Z95 * s)), 1), hi95=round(float(np.exp(m + Z95 * s)), 1))
              for d, m, s in zip(idx, mean, sd)]
    return PredictResponse(model_version=meta["version"], origin_month=f"{last:%Y-%m}",
                           horizon=req.horizon, forecast=points)


@app.post("/reload")
def reload_model(x_reload_token: str = Header(...)):
    """Called by Django after it promotes a new model version."""
    if x_reload_token != RELOAD_TOKEN:
        raise HTTPException(403, "Invalid token")
    load_artifact()
    return {"status": "reloaded", "model_version": state.meta.get("version")}
