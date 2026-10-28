import json

import pytest
from fastapi.testclient import TestClient

from src.api import main

FEATURES = [
    "dayofyear",
    "doy_sin",
    "doy_cos",
    "lag_1",
    "lag_2",
    "lag_3",
    "lag_7",
    "lag_14",
    "roll_mean_7",
    "roll_mean_30",
]


@pytest.fixture(autouse=True)
def request_log(tmp_path, monkeypatch):
    path = tmp_path / "requests.jsonl"
    monkeypatch.setattr(main, "REQUEST_LOG", path)
    return path


def test_health_ok(monkeypatch):
    monkeypatch.setattr(main, "_bundle", {"model": object(), "features": FEATURES})
    client = TestClient(main.app)

    r = client.get("/health")

    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_health_without_model_returns_503(monkeypatch):
    monkeypatch.setattr(main, "_bundle", None)
    client = TestClient(main.app)

    r = client.get("/health")

    assert r.status_code == 503
    assert r.json()["status"] == "model_missing"
    assert r.json()["model_loaded"] is False


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


def test_predict_appends_features_to_request_log(monkeypatch, request_log):
    class Stub:
        def predict(self, X):
            return [21.5]

    monkeypatch.setattr(main, "_bundle", {"model": Stub(), "features": FEATURES})
    client = TestClient(main.app)
    body = {"date": "1991-01-01", "recent_temps": [10.0] * 30}

    assert client.post("/predict", json=body).status_code == 200
    assert client.post("/predict", json=body).status_code == 200

    records = [json.loads(line) for line in request_log.read_text().splitlines()]
    assert len(records) == 2
    assert set(FEATURES) <= set(records[0])
    assert records[0]["lag_1"] == 10.0
    assert records[0]["prediction"] == 21.5
    assert "ts" in records[0]


VALID_TEMPS = "[" + ", ".join(["10.0"] * 30) + "]"


def raw_body(date='"1991-01-01"', temps=VALID_TEMPS):
    """Request body as raw JSON text, so NaN and Infinity reach the server unchanged."""
    return f'{{"date": {date}, "recent_temps": {temps}}}'


def temps_with(value, n=30):
    return "[" + ", ".join([value] + ["10.0"] * (n - 1)) + "]"


@pytest.mark.parametrize(
    "body, status",
    [
        (raw_body(), 200),
        (raw_body(temps=temps_with("10.0", n=29)), 422),
        (raw_body(temps=temps_with("10.0", n=367)), 422),
        (raw_body(temps=temps_with("61")), 422),
        (raw_body(temps=temps_with("-51")), 422),
        (raw_body(temps=temps_with("NaN")), 422),
        (raw_body(temps=temps_with("Infinity")), 422),
        (raw_body(date='"01/01/1991"'), 422),
        (raw_body(date='"1991-01-01T00:00:00"'), 422),
        (raw_body(date='"1991-02-30"'), 422),
        (raw_body(date="662688000"), 422),
    ],
    ids=[
        "valid",
        "29-values",
        "367-values",
        "temp-61",
        "temp-minus-51",
        "nan",
        "infinity",
        "us-date",
        "datetime-string",
        "impossible-date",
        "integer-date",
    ],
)
def test_predict_input_validation(monkeypatch, body, status):
    class Stub:
        def predict(self, X):
            return [21.5]

    monkeypatch.setattr(main, "_bundle", {"model": Stub(), "features": FEATURES})
    client = TestClient(main.app)

    r = client.post("/predict", content=body, headers={"Content-Type": "application/json"})

    assert r.status_code == status, r.text
    if status == 422:
        assert isinstance(r.json()["detail"], list)


def test_predict_response_contract_unchanged(monkeypatch):
    class Stub:
        def predict(self, X):
            return [21.4567]

    monkeypatch.setattr(main, "_bundle", {"model": Stub(), "features": FEATURES})
    client = TestClient(main.app)
    body = {"date": "1991-01-01", "recent_temps": [10.0] * 366}

    r = client.post("/predict", json=body)

    assert r.status_code == 200
    assert r.json() == {"date": "1991-01-01", "prediction": 21.46}


def test_health_reports_git_sha(monkeypatch):
    monkeypatch.setattr(main, "_bundle", {"model": object(), "features": FEATURES})
    monkeypatch.delenv("GIT_SHA", raising=False)
    monkeypatch.setenv("RENDER_GIT_COMMIT", "abc1234")
    client = TestClient(main.app)

    assert client.get("/health").json()["git_sha"] == "abc1234"

    # An explicit build arg wins over Render's variable.
    monkeypatch.setenv("GIT_SHA", "def5678")
    assert client.get("/health").json()["git_sha"] == "def5678"


def test_health_git_sha_unknown_without_env(monkeypatch):
    monkeypatch.delenv("GIT_SHA", raising=False)
    monkeypatch.delenv("RENDER_GIT_COMMIT", raising=False)
    client = TestClient(main.app)

    assert client.get("/health").json()["git_sha"] == "unknown"
