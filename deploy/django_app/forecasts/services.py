"""Model registry access, cached inference and retraining orchestration."""
from __future__ import annotations

import hashlib
import logging
import pickle
from functools import lru_cache
from pathlib import Path

import pandas as pd
from django.conf import settings
from django.db import transaction
from django.utils import timezone

from . import ml
from .models import Forecast, ModelVersion, PriceObservation

log = logging.getLogger(__name__)


@lru_cache(maxsize=4)
def _load(path: str, sha256: str):
    data = Path(path).read_bytes()
    if hashlib.sha256(data).hexdigest() != sha256:
        raise RuntimeError(f"Artifact hash mismatch for {path}")
    return pickle.loads(data)


def production_model() -> tuple[ModelVersion, object]:
    mv = ModelVersion.objects.get(status="production")
    return mv, _load(mv.artifact_path, mv.artifact_sha256)  # loaded once per process, then cached


def predict(horizon: int = 12) -> tuple[ModelVersion, pd.DataFrame]:
    mv, res = production_model()
    return mv, ml.forecast_frame(res, pd.Timestamp(mv.trained_through), horizon)


def price_series() -> pd.Series:
    rows = PriceObservation.objects.values_list("month", "price")
    s = pd.Series({pd.Timestamp(m): p for m, p in rows}).sort_index()
    s.index.freq = "MS"
    return s.astype(float)


def retrain(promote: bool = True) -> tuple[ModelVersion, str]:
    s = price_series()
    full = pd.date_range(s.index.min(), s.index.max(), freq="MS")
    if len(full) != len(s):
        raise ValueError(f"{len(full) - len(s)} month(s) missing; fix the data before retraining")

    y, flags = ml.clean(s)
    for month, flag in flags.items():  # keep data-quality flags visible in the admin
        PriceObservation.objects.filter(month=month.date()).update(quality_flag=flag)

    checks = ml.holdout_check(y, s)
    res = ml.fit(y)
    version = f"{s.index[-1]:%Y-%m}_{timezone.now():%Y%m%d-%H%M%S}"
    settings.ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    path = settings.ARTIFACT_DIR / f"arima_{version}.pkl"
    blob = pickle.dumps(res)
    path.write_bytes(blob)

    current = ModelVersion.objects.filter(status="production").first()
    ok, reason = ml.passes_quality_gate(checks, current and {"holdout_mape": current.holdout_mape})

    with transaction.atomic():
        mv = ModelVersion.objects.create(
            version=version, artifact_path=str(path), artifact_sha256=hashlib.sha256(blob).hexdigest(),
            trained_through=s.index[-1].date(), n_obs=len(s), status="candidate", **checks)
        fc = ml.forecast_frame(res, s.index[-1], 12)
        Forecast.objects.bulk_create([Forecast(model_version=mv, month=d.date(), **r.to_dict())
                                      for d, r in fc.iterrows()])
        if not ok:
            mv.status = "rejected"
            mv.save(update_fields=["status"])
        elif promote:
            ModelVersion.objects.filter(status="production").update(status="retired")
            mv.status = "production"
            mv.save(update_fields=["status"])
            transaction.on_commit(notify_inference_service)
    return mv, reason


def notify_inference_service() -> None:
    """Tell the FastAPI service to hot-swap the model (no-op if not configured)."""
    if not settings.FASTAPI_RELOAD_URL:
        return
    import httpx
    try:
        httpx.post(settings.FASTAPI_RELOAD_URL, headers={"x-reload-token": settings.FASTAPI_RELOAD_TOKEN},
                   timeout=5).raise_for_status()
    except httpx.HTTPError as exc:  # don't fail the retrain; FastAPI also reloads on restart
        log.warning("FastAPI reload failed: %s", exc)
