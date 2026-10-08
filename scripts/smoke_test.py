"""Post-deploy smoke test: prove the live service answers on its critical paths.

Usage: python scripts/smoke_test.py https://<service>.onrender.com [expected_git_sha]
Defaults to http://localhost:8000. With expected_git_sha (full or short), the test also
fails unless /health reports that commit, i.e. the deploy you expect is the one serving.
Timeouts are generous because a free-tier instance cold-starts after idling.
"""

import sys

import requests

TIMEOUT = 90  # seconds; covers a Render free-tier cold start


def main(base_url: str, expected_sha: str | None = None) -> None:
    base_url = base_url.rstrip("/")

    health = requests.get(f"{base_url}/health", timeout=TIMEOUT)
    health.raise_for_status()
    body = health.json()
    print("health:", body)
    assert body["status"] == "ok", "health status is not ok"
    assert body["model_loaded"] is True, "model is not loaded on the server"
    if expected_sha:
        live = body.get("git_sha", "missing")
        # Either side may be a short sha, so compare on the shorter of the two.
        n = min(len(live), len(expected_sha))
        matches = n >= 7 and live[:n] == expected_sha[:n]
        assert matches, f"server is running {live}, expected {expected_sha}"

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
    main(base, sys.argv[2] if len(sys.argv) > 2 else None)
