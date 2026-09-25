"""Post-deploy smoke test: prove the live service answers on its critical paths.

Usage: python scripts/smoke_test.py https://<service>.onrender.com
Defaults to http://localhost:8000. Timeouts are generous because a free-tier
instance cold-starts after idling.
"""

import sys

import requests

TIMEOUT = 90  # seconds; covers a Render free-tier cold start


def main(base_url: str) -> None:
    base_url = base_url.rstrip("/")

    health = requests.get(f"{base_url}/health", timeout=TIMEOUT)
    health.raise_for_status()
    body = health.json()
    print("health:", body)
    assert body["status"] == "ok", "health status is not ok"
    assert body["model_loaded"] is True, "model is not loaded on the server"

    payload = {"date": "1991-01-01", "recent_temps": [12.0] * 30}
    pred = requests.post(f"{base_url}/predict", json=payload, timeout=TIMEOUT)
    pred.raise_for_status()
    result = pred.json()
    print("predict:", result)
    assert isinstance(result["prediction"], float), "prediction is not a number"

    metrics = requests.get(f"{base_url}/metrics", timeout=TIMEOUT)
    metrics.raise_for_status()
    assert "predictions_total" in metrics.text, "metrics endpoint missing counter"
    print("metrics: ok")

    print(f"SMOKE TEST PASSED for {base_url}")


if __name__ == "__main__":
    base = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
    main(base)
