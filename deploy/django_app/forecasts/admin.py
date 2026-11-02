from django.contrib import admin

from .models import Forecast, ModelVersion, PriceObservation


@admin.register(PriceObservation)
class PriceAdmin(admin.ModelAdmin):
    list_display = ("month", "price", "quality_flag", "source")
    list_filter = ("quality_flag", "source")


class ForecastInline(admin.TabularInline):
    model = Forecast
    extra = 0
    readonly_fields = ("month", "mean", "lo80", "hi80", "lo95", "hi95")


@admin.register(ModelVersion)
class ModelVersionAdmin(admin.ModelAdmin):
    list_display = ("version", "status", "trained_through", "holdout_mape", "naive_holdout_mape", "holdout_coverage95")
    list_filter = ("status",)
    inlines = [ForecastInline]
