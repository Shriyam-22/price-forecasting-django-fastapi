# Monthly Price Forecasting: Data Cleaning, Time-Series Modelling and Django/FastAPI Deployment

This project forecasts an agricultural commodity's average monthly price 12 months ahead. It covers the full workflow: data-quality forensics, a rigorous model comparison and a production-style serving stack with **Django REST Framework** and **FastAPI**.

Built as the technical assignment for the AI/ML Developer role at Way2Agribusiness India Pvt. Ltd.

## Highlights

- **Found a hidden copy-paste error in the data.** 30 months (Jun 2021 – Nov 2023) are an exact, value-for-value copy of 2007–2010 prices. A self-matching search detects it automatically, and the months are re-estimated with a seasonally adjusted log interpolation. Leaving them in inflates estimated volatility by about 70% and hides the momentum effect.
- **Compared 10 model families and 71 tuned SARIMA configurations** (naive baselines, ETS, Theta, ARIMA/SARIMA, LightGBM) using a 13-origin rolling backtest plus an untouched 12-month holdout.
- **Selected model: ARIMA(1,1,0) on log prices.** Holdout MAPE **8.7%** (best of all candidates), MASE 0.71, and every holdout month inside the 95% prediction interval. LightGBM reached about 16%: with about 250 points and no exogenous drivers, ML overfits.
- **Production stack:**
  - **Django + DRF**: price ingestion, data-quality flags, model registry (`candidate → production → retired`), champion/challenger quality gate, monthly `retrain_forecast` command, forecast API.
  - **FastAPI**: low-latency `/predict` with **real-time state updates** (new prices update the forecast without a refit), Pydantic validation, hot `/reload` triggered by Django.
  - 13 automated tests across both services, including checks that the APIs reproduce the notebook forecast exactly.

## Repository layout

```
Price_Forecasting_Shriyam.ipynb     # full analysis and answers (executed)
price_data.csv                      # input data
artifacts/                          # trained model, metadata and forecast, cleaned data with flags
deploy/
  django_app/                       # Django + DRF project (system of record, retraining, registry)
  fastapi_service/                  # FastAPI inference microservice
build_notebook.py                   # generates the notebook from source
requirements.txt
```

## Run it

```bash
pip install -r requirements.txt

# 1. Notebook (about 2 minutes): writes artifacts/
jupyter nbconvert --to notebook --execute Price_Forecasting_Shriyam.ipynb

# 2. FastAPI inference service
cd deploy/fastapi_service && uvicorn main:app --reload        # http://127.0.0.1:8000/docs
pytest -q

# 3. Django API
cd deploy/django_app
python manage.py migrate
python manage.py load_prices ../../price_data.csv
python manage.py retrain_forecast       # clean → fit → quality gate → promote
python manage.py runserver               # GET http://127.0.0.1:8000/api/v1/forecast/?horizon=12
python manage.py test forecasts
```

### Example: real-time prediction with a newer price

```bash
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" \
  -d '{"horizon": 3, "recent_prices": [{"month": "2025-10-01", "price": 17000}]}'
```

## Results

| Evaluation | MAPE | MASE | 95% interval coverage |
|---|---|---|---|
| Training, 1-step in-sample | 4.2% | – | – |
| Holdout, Oct 2024 – Sep 2025 | 8.7% | 0.71 | 100% |
| Rolling backtest, 13 × 12 months | 10.5% | 0.55 | 99% |

**Forecast, Oct 2025 – Sep 2026:** about 15,800, essentially flat, with an 80% range of about 11,800–21,200 by Sep 2026.

**Tech:** Python, pandas, statsmodels, scikit-learn, LightGBM, matplotlib, Django, Django REST Framework, FastAPI, Pydantic, pytest.

---
Shriyam · shriyamgopalgupta@gmail.com · [github.com/Shriyam-22](https://github.com/Shriyam-22)
