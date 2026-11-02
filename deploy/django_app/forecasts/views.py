from rest_framework import generics, permissions, serializers, status
from rest_framework.response import Response
from rest_framework.views import APIView

from . import services
from .models import ModelVersion, PriceObservation


class PriceSerializer(serializers.ModelSerializer):
    class Meta:
        model = PriceObservation
        fields = ["month", "price", "source", "quality_flag"]
        read_only_fields = ["quality_flag"]

    def validate_month(self, v):
        return v.replace(day=1)

    def validate_price(self, v):
        if v <= 0:
            raise serializers.ValidationError("Price must be positive")
        return v


class ModelVersionSerializer(serializers.ModelSerializer):
    class Meta:
        model = ModelVersion
        exclude = ["artifact_path", "artifact_sha256", "id"]


class PriceListCreate(generics.ListCreateAPIView):
    """GET: price history. POST (token auth): add a new monthly price."""
    queryset = PriceObservation.objects.all()
    serializer_class = PriceSerializer


class ForecastView(APIView):
    """GET /api/v1/forecast/?horizon=12"""
    permission_classes = [permissions.AllowAny]

    def get(self, request):
        try:
            horizon = int(request.query_params.get("horizon", 12))
        except ValueError:
            return Response({"detail": "horizon must be an integer"}, status=400)
        if not 1 <= horizon <= 24:
            return Response({"detail": "horizon must be between 1 and 24"}, status=400)
        try:
            mv, fc = services.predict(horizon)
        except ModelVersion.DoesNotExist:
            return Response({"detail": "No production model yet"}, status=status.HTTP_503_SERVICE_UNAVAILABLE)
        return Response({
            "model_version": mv.version,
            "trained_through": f"{mv.trained_through:%Y-%m}",
            "forecast": [{"month": f"{d:%Y-%m}", **{k: round(v, 1) for k, v in r.items()}} for d, r in fc.iterrows()],
        })


class ModelInfoView(APIView):
    permission_classes = [permissions.AllowAny]

    def get(self, request):
        mv = ModelVersion.objects.filter(status="production").first()
        if mv is None:
            return Response({"detail": "No production model yet"}, status=503)
        return Response(ModelVersionSerializer(mv).data)


class RetrainView(APIView):
    """POST /api/v1/retrain/ (admin only). In production this runs as a Celery task."""
    permission_classes = [permissions.IsAdminUser]

    def post(self, request):
        mv, reason = services.retrain()
        return Response({"version": mv.version, "status": mv.status, "gate": reason,
                         "holdout_mape": round(mv.holdout_mape, 2)}, status=201)
