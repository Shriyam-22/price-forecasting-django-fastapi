import pandas as pd
from django.core.management.base import BaseCommand

from forecasts.models import PriceObservation


class Command(BaseCommand):
    help = "Load monthly prices from a CSV with columns date, avg_monthly_price"

    def add_arguments(self, parser):
        parser.add_argument("csv_path")
        parser.add_argument("--source", default="csv")

    def handle(self, csv_path, source, **opts):
        df = pd.read_csv(csv_path, parse_dates=["date"])
        n = 0
        for d, p in zip(df["date"], df["avg_monthly_price"]):
            PriceObservation.objects.update_or_create(month=d.date().replace(day=1),
                                                      defaults={"price": float(p), "source": source})
            n += 1
        self.stdout.write(self.style.SUCCESS(f"Loaded {n} monthly prices"))
