import json
from pathlib import Path

from fastapi.testclient import TestClient

from main import app, ARTIFACT_DIR

META = json.loads((ARTIFACT_DIR / "model_meta.json").read_text())


def client():
    return TestClient(app)


def test_health():
    with client() as c:
        r = c.get("/health")
        assert r.status_code == 200
        assert r.json()["model_version"] == META["version"]


def test_predict_matches_notebook_forecast():
    with client() as c:
        r = c.post("/predict", json={"horizon": 12})
        assert r.status_code == 200
        body = r.json()
        assert body["origin_month"] == META["trained_through"]
        assert len(body["forecast"]) == 12
        for got, exp in zip(body["forecast"], META["forecast"]):
            assert got["month"] == exp["month"]
            assert abs(got["mean"] - exp["mean"]) < 1.0
            assert abs(got["hi95"] - exp["hi95"]) < 1.0
            assert got["lo95"] < got["lo80"] < got["mean"] < got["hi80"] < got["hi95"]


def test_realtime_update_moves_origin():
    with client() as c:
        r = c.post("/predict", json={"horizon": 3, "recent_prices": [{"month": "2025-10-01", "price": 17000}]})
        assert r.status_code == 200
        body = r.json()
        assert body["origin_month"] == "2025-10"
        assert body["forecast"][0]["month"] == "2025-11"
        assert body["forecast"][0]["mean"] > META["forecast"][0]["mean"]  # starts from the newer, higher price


def test_rejects_gap_in_recent_prices():
    with client() as c:
        r = c.post("/predict", json={"recent_prices": [{"month": "2025-12-01", "price": 15000}]})
        assert r.status_code == 422


def test_validation():
    with client() as c:
        assert c.post("/predict", json={"horizon": 0}).status_code == 422
        assert c.post("/predict", json={"horizon": 30}).status_code == 422


def test_reload_requires_token():
    with client() as c:
        assert c.post("/reload", headers={"x-reload-token": "wrong"}).status_code == 403
        assert c.post("/reload", headers={"x-reload-token": "change-me"}).status_code == 200
