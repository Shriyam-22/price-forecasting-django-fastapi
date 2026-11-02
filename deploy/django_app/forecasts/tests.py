import json
import tempfile
from pathlib import Path

from django.contrib.auth.models import User
from django.core.management import call_command
from django.test import TestCase, override_settings
from rest_framework.authtoken.models import Token
from rest_framework.test import APIClient

from .models import ModelVersion, PriceObservation

DATA = Path(__file__).resolve().parents[3] / "price_data.csv"
NOTEBOOK_META = Path(__file__).resolve().parents[3] / "artifacts" / "model_meta.json"


@override_settings(ARTIFACT_DIR=Path(tempfile.mkdtemp()))
class PipelineTests(TestCase):
    @classmethod
    def setUpTestData(cls):
        call_command("load_prices", str(DATA))
        call_command("retrain_forecast")

    def setUp(self):
        self.api = APIClient()

    def test_data_quality_flags(self):
        self.assertEqual(PriceObservation.objects.count(), 249)
        self.assertEqual(PriceObservation.objects.filter(quality_flag="copied_block").count(), 30)
        self.assertEqual(PriceObservation.objects.filter(quality_flag="outlier").count(), 2)

    def test_model_promoted_and_beats_benchmark(self):
        mv = ModelVersion.objects.get(status="production")
        self.assertLess(mv.holdout_mape, mv.naive_holdout_mape)
        self.assertAlmostEqual(mv.holdout_mape, 8.71, places=1)
        self.assertEqual(mv.forecasts.count(), 12)

    def test_forecast_endpoint_matches_notebook(self):
        r = self.api.get("/api/v1/forecast/?horizon=12")
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertEqual(body["trained_through"], "2025-09")
        expected = json.loads(NOTEBOOK_META.read_text())["forecast"]
        for got, exp in zip(body["forecast"], expected):
            self.assertEqual(got["month"], exp["month"])
            self.assertAlmostEqual(got["mean"], exp["mean"], delta=1.0)

    def test_horizon_validation(self):
        self.assertEqual(self.api.get("/api/v1/forecast/?horizon=0").status_code, 400)
        self.assertEqual(self.api.get("/api/v1/forecast/?horizon=abc").status_code, 400)

    def test_writes_need_auth(self):
        payload = {"month": "2025-10-01", "price": 16100}
        self.assertEqual(self.api.post("/api/v1/prices/", payload).status_code, 401)
        user = User.objects.create_user("ops", password="x")
        self.api.credentials(HTTP_AUTHORIZATION=f"Token {Token.objects.create(user=user).key}")
        self.assertEqual(self.api.post("/api/v1/prices/", payload).status_code, 201)
        self.assertEqual(self.api.post("/api/v1/prices/", {"month": "2025-11-01", "price": -5}).status_code, 400)

    def test_retrain_with_new_month_rolls_model_forward(self):
        PriceObservation.objects.create(month="2025-10-01", price=16100)
        call_command("retrain_forecast")
        self.assertEqual(ModelVersion.objects.filter(status="production").count(), 1)
        self.assertEqual(ModelVersion.objects.filter(status="retired").count(), 1)
        r = self.api.get("/api/v1/model/").json()
        self.assertEqual(r["trained_through"], "2025-10-01")
        self.assertEqual(self.api.get("/api/v1/forecast/?horizon=1").json()["forecast"][0]["month"], "2025-11")

    def test_retrain_refuses_gaps(self):
        PriceObservation.objects.create(month="2025-12-01", price=16000)
        with self.assertRaises(ValueError):
            call_command("retrain_forecast")
