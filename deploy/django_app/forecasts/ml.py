"""Training pipeline used by the retrain_forecast command (same logic as the notebook)."""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

ORDER = (1, 1, 0)
Z80, Z95 = 1.2816, 1.96


def find_copied_months(s: pd.Series, min_run: int = 6) -> pd.DatetimeIndex:
    v, n, out = s.values, len(s), set()
    for lag in range(1, n):
        run = 0
        for i in range(lag, n + 1):
            if i < n and v[i] == v[i - lag]:
                run += 1
                continue
            if run >= min_run:
                out.update(s.index[i - run:i])
            run = 0
    return pd.DatetimeIndex(sorted(out))


def find_outliers(s: pd.Series, window: int = 7, threshold: float = 0.20) -> pd.DatetimeIndex:
    x = np.log(s)
    dev = x - x.rolling(window, center=True, min_periods=window // 2 + 1).median()
    return dev[dev.abs() > threshold].index


def clean(s: pd.Series) -> tuple[pd.Series, pd.Series]:
    copied = find_copied_months(s)
    outliers = find_outliers(s.drop(copied))
    flags = pd.Series("ok", index=s.index)
    flags[s.diff() == 0] = "stale"
    flags[copied] = "copied_block"
    flags[outliers] = "outlier"
    x = np.log(s.copy())
    x[flags.isin(["copied_block", "outlier"])] = np.nan
    trend = x.rolling(13, center=True, min_periods=9).mean()
    detr = (x - trend).dropna()
    seasonal = detr.groupby(detr.index.month).median()
    seasonal -= seasonal.mean()
    seas = pd.Series(x.index.month.map(seasonal).values, index=x.index)
    filled = np.exp((x - seas).interpolate(method="time", limit_direction="both") + seas)
    return np.exp(x).where(x.notna(), filled), flags


def fit(y: pd.Series):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return SARIMAX(np.log(y), order=ORDER, enforce_stationarity=False,
                       enforce_invertibility=False).fit(disp=False)


def forecast_frame(res, last_month: pd.Timestamp, h: int = 12) -> pd.DataFrame:
    f = res.get_forecast(h)
    m = np.asarray(f.predicted_mean)
    sd = np.sqrt(np.asarray(f.var_pred_mean))
    idx = pd.date_range(last_month + pd.offsets.MonthBegin(1), periods=h, freq="MS")
    return pd.DataFrame({"mean": np.exp(m), "lo80": np.exp(m - Z80 * sd), "hi80": np.exp(m + Z80 * sd),
                         "lo95": np.exp(m - Z95 * sd), "hi95": np.exp(m + Z95 * sd)}, index=idx)


def holdout_check(y_clean: pd.Series, y_raw: pd.Series, h: int = 12) -> dict:
    """Refit without the last h months and score against the real values."""
    train, actual = y_clean[:-h], y_raw[-h:].values
    fc = forecast_frame(fit(train), train.index[-1], h)
    mape = float(np.mean(np.abs(fc["mean"].values - actual) / actual) * 100)
    naive = float(np.mean(np.abs(train.iloc[-1] - actual) / actual) * 100)
    cover = float(np.mean((actual >= fc["lo95"].values) & (actual <= fc["hi95"].values)) * 100)
    return {"holdout_mape": mape, "naive_holdout_mape": naive, "holdout_coverage95": cover}


def passes_quality_gate(m: dict, production: dict | None) -> tuple[bool, str]:
    if m["holdout_mape"] > 1.10 * m["naive_holdout_mape"]:
        return False, "worse than the random-walk benchmark by more than 10%"
    if not 75 <= m["holdout_coverage95"] <= 100:
        return False, "95% interval coverage outside 75-100%"
    if production and m["holdout_mape"] > 1.20 * production["holdout_mape"]:
        return False, "more than 20% worse than the current production model"
    return True, "passed"
