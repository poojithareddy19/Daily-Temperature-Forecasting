import hashlib
import json
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path

import joblib
import yaml
from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import FileResponse
from prometheus_client import Counter, Histogram, make_asgi_app

from src.api.schemas import PredictionRequest, PredictionResponse
from src.config import PROJECT_ROOT, load_config
from src.features.serving import features_from_history
from src.logger import get_logger

logger = get_logger(__name__)

cfg = load_config()
with open(PROJECT_ROOT / "params.yaml") as f:
    FEATURE_PARAMS = yaml.safe_load(f)["features"]

REQUEST_LOG = PROJECT_ROOT / cfg["paths"]["request_log"]

_bundle = None
_model_sha256 = None


def _load_model():
    global _bundle, _model_sha256

    path = PROJECT_ROOT / cfg["paths"]["model_path"]

    if path.exists():
        _bundle = joblib.load(path)
        _model_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        logger.info("Loaded model from %s (sha256 %s)", path, _model_sha256)
    else:
        logger.warning("No model at %s; /predict returns 503", path)


@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_model()
    yield


app = FastAPI(title="Temperature Forecast API", version="1.0.0", lifespan=lifespan)

STATIC_DIR = Path(__file__).resolve().parent / "static"

# Prometheus scrape endpoint and the metrics it exposes. Defined once at import
# time: registering a metric inside a request handler raises on the second call.
app.mount("/metrics", make_asgi_app())
PREDICTIONS = Counter("predictions_total", "Total prediction requests")
PRED_LATENCY = Histogram("prediction_latency_seconds", "Prediction latency in seconds")


@app.get("/", include_in_schema=False)
def root():
    """Serve the single-page UI."""
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health(response: Response):
    # 503 without a model, so Render does not route traffic to a service that cannot predict.
    if _bundle is None:
        response.status_code = 503
    return {
        "status": "ok" if _bundle is not None else "model_missing",
        "model_loaded": _bundle is not None,
        "model_sha256": _model_sha256,
    }


def _serving_features(req: PredictionRequest) -> dict:
    try:
        d = datetime.strptime(req.date, "%Y-%m-%d")
    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail="date must be YYYY-MM-DD",
        ) from exc

    return features_from_history(
        d, req.recent_temps, FEATURE_PARAMS["lags"], FEATURE_PARAMS["roll_windows"]
    )


def _log_request(feat: dict, prediction: float) -> None:
    """Append the request's features to a JSONL file; the drift report reads it back."""
    record = {"ts": datetime.now(UTC).isoformat(), **feat, "prediction": prediction}
    try:
        REQUEST_LOG.parent.mkdir(parents=True, exist_ok=True)
        with open(REQUEST_LOG, "a") as f:
            f.write(json.dumps(record) + "\n")
    except OSError:
        # Losing a log line must never fail the prediction itself.
        logger.warning("Could not write request log to %s", REQUEST_LOG, exc_info=True)


@app.post("/predict", response_model=PredictionResponse)
def predict(req: PredictionRequest):
    if _bundle is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded",
        )

    feat = _serving_features(req)
    with PRED_LATENCY.time():
        value = float(_bundle["model"].predict([[feat[c] for c in _bundle["features"]]])[0])
    PREDICTIONS.inc()
    _log_request(feat, value)

    logger.info(
        "Predicted %.2f for date=%s",
        value,
        req.date,
    )

    return PredictionResponse(
        date=req.date,
        prediction=round(value, 2),
    )
