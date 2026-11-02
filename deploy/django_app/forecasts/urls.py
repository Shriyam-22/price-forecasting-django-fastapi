from django.urls import path

from . import views

urlpatterns = [
    path("prices/", views.PriceListCreate.as_view(), name="prices"),
    path("forecast/", views.ForecastView.as_view(), name="forecast"),
    path("model/", views.ModelInfoView.as_view(), name="model-info"),
    path("retrain/", views.RetrainView.as_view(), name="retrain"),
]
