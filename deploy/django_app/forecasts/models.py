from django.db import models


class PriceObservation(models.Model):
    QUALITY = [("ok", "ok"), ("copied_block", "copied block"), ("outlier", "outlier"), ("stale", "stale")]
    month = models.DateField(unique=True, help_text="First day of the month")
    price = models.FloatField()
    source = models.CharField(max_length=50, default="manual")
    quality_flag = models.CharField(max_length=20, choices=QUALITY, default="ok")
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["month"]

    def __str__(self):
        return f"{self.month:%Y-%m}: {self.price:,.0f}"


class ModelVersion(models.Model):
    STATUS = [("candidate", "candidate"), ("production", "production"), ("retired", "retired"),
              ("rejected", "rejected")]
    version = models.CharField(max_length=40, unique=True)
    algorithm = models.CharField(max_length=60, default="ARIMA(1,1,0) on log price")
    artifact_path = models.CharField(max_length=255)
    artifact_sha256 = models.CharField(max_length=64)
    trained_through = models.DateField()
    n_obs = models.PositiveIntegerField()
    holdout_mape = models.FloatField(help_text="MAPE % on the last 12 months, fitted without them")
    naive_holdout_mape = models.FloatField(help_text="Benchmark: random-walk MAPE % on the same months")
    holdout_coverage95 = models.FloatField()
    status = models.CharField(max_length=12, choices=STATUS, default="candidate")
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]
        constraints = [models.UniqueConstraint(fields=["status"], condition=models.Q(status="production"),
                                               name="single_production_model")]

    def __str__(self):
        return f"{self.version} ({self.status})"


class Forecast(models.Model):
    model_version = models.ForeignKey(ModelVersion, on_delete=models.CASCADE, related_name="forecasts")
    month = models.DateField()
    mean = models.FloatField()
    lo80 = models.FloatField()
    hi80 = models.FloatField()
    lo95 = models.FloatField()
    hi95 = models.FloatField()

    class Meta:
        ordering = ["month"]
        unique_together = ("model_version", "month")
