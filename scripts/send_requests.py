"""Send a batch of /predict requests so the Grafana panels have traffic to show.

Usage: python scripts/send_requests.py [base_url] [count]
Defaults to http://localhost:8000 and 200 requests. Each request uses a random
30-day window of the real series, so the logged inputs look like real traffic.
"""

import random
import sys
import time
from pathlib import Path

import pandas as pd
import requests

RAW = Path(__file__).resolve().parents[1] / "data" / "raw" / "daily-min-temperatures.csv"


def main(base_url: str, count: int) -> None:
    temps = pd.read_csv(RAW).iloc[:, 1].astype(float).tolist()
    dates = pd.read_csv(RAW).iloc[:, 0].tolist()
    ok = 0
    for _ in range(count):
        i = random.randrange(30, len(temps))
        body = {"date": dates[i], "recent_temps": temps[i - 30 : i][::-1]}
        r = requests.post(f"{base_url.rstrip('/')}/predict", json=body, timeout=30)
        ok += r.status_code == 200
        time.sleep(random.uniform(0.05, 0.5))
    print(f"{ok}/{count} requests succeeded")


if __name__ == "__main__":
    base = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 200
    main(base, n)
