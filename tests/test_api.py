from fastapi.testclient import TestClient

from src.api import main

FEATURES = [
    "month",
    "dayofyear",
    "dayofweek",
    "lag_1",
    "lag_2",
    "lag_3",
    "lag_7",
    "lag_14",
    "roll_mean_7",
    "roll_mean_30",
]


def test_health_ok():
    client = TestClient(main.app)

    r = client.get("/health")

    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_predict_without_model_returns_503(monkeypatch):
    monkeypatch.setattr(main, "_bundle", None)

    client = TestClient(main.app)

    body = {
        "date": "1991-01-01",
        "recent_temps": [10.0] * 30,
    }

    assert client.post("/predict", json=body).status_code == 503


def test_predict_with_stub_model(monkeypatch):
    class Stub:
        def predict(self, X):
            return [21.5]

    monkeypatch.setattr(
        main,
        "_bundle",
        {
            "model": Stub(),
            "features": FEATURES,
        },
    )

    client = TestClient(main.app)

    body = {
        "date": "1991-01-01",
        "recent_temps": [10.0] * 30,
    }

    r = client.post("/predict", json=body)

    assert r.status_code == 200
    assert r.json()["prediction"] == 21.5


def test_metrics_endpoint_reports_predictions(monkeypatch):
    class Stub:
        def predict(self, X):
            return [21.5]

    monkeypatch.setattr(main, "_bundle", {"model": Stub(), "features": FEATURES})
    client = TestClient(main.app)
    body = {"date": "1991-01-01", "recent_temps": [10.0] * 30}
    assert client.post("/predict", json=body).status_code == 200

    r = client.get("/metrics")
    assert r.status_code == 200
    assert "predictions_total" in r.text
    assert "prediction_latency_seconds_bucket" in r.text


def test_root_serves_ui():
    client = TestClient(main.app)
    r = client.get("/")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/html")
    assert "Predict" in r.text
