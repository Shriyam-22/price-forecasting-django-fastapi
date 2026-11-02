from django.core.management.base import BaseCommand

from forecasts import services


class Command(BaseCommand):
    help = "Clean data, refit the model, run the quality gate and promote it if it passes (schedule monthly)."

    def add_arguments(self, parser):
        parser.add_argument("--no-promote", action="store_true", help="Register as candidate only")

    def handle(self, no_promote, **opts):
        mv, reason = services.retrain(promote=not no_promote)
        self.stdout.write(self.style.SUCCESS(
            f"{mv.version}: status={mv.status} gate={reason} holdout MAPE={mv.holdout_mape:.2f}% "
            f"(naive {mv.naive_holdout_mape:.2f}%, 95% coverage {mv.holdout_coverage95:.0f}%)"))
