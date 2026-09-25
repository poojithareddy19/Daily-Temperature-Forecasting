from datetime import datetime

import joblib
from fastapi import FastAPI, HTTPException
from prometheus_client import Counter, Histogram, make_asgi_app

from src.api.schemas import PredictionRequest, PredictionResponse
from src.config import PROJECT_ROOT, load_config
from src.logger import get_logger

logger = get_logger(__name__)

cfg = load_config()

app = FastAPI(title="Temperature Forecast API", version="1.0.0")

# Prometheus scrape endpoint and the metrics it exposes. Defined once at import
# time: registering a metric inside a request handler raises on the second call.
app.mount("/metrics", make_asgi_app())
PREDICTIONS = Counter("predictions_total", "Total prediction requests")
PRED_LATENCY = Histogram("prediction_latency_seconds", "Prediction latency in seconds")

_bundle = None


@app.on_event("startup")
def _load_model():
    global _bundle

    path = PROJECT_ROOT / cfg["paths"]["model_path"]

    if path.exists():
        _bundle = joblib.load(path)
        logger.info("Loaded model from %s", path)
    else:
        logger.warning("No model at %s; /predict returns 503", path)


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _bundle is not None}


def _serving_features(req: PredictionRequest) -> list:
    t = req.recent_temps

    try:
        d = datetime.strptime(req.date, "%Y-%m-%d")
    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail="date must be YYYY-MM-DD",
        ) from exc

    feat = {
        "month": d.month,
        "dayofyear": d.timetuple().tm_yday,
        "dayofweek": d.weekday(),
        "lag_1": t[0],
        "lag_2": t[1],
        "lag_3": t[2],
        "lag_7": t[6],
        "lag_14": t[13],
        "roll_mean_7": sum(t[:7]) / 7,
        "roll_mean_30": sum(t[:30]) / 30,
    }

    return [feat[c] for c in _bundle["features"]]


@app.post("/predict", response_model=PredictionResponse)
def predict(req: PredictionRequest):
    if _bundle is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded",
        )

    with PRED_LATENCY.time():
        value = float(_bundle["model"].predict([_serving_features(req)])[0])
    PREDICTIONS.inc()

    logger.info(
        "Predicted %.2f for date=%s",
        value,
        req.date,
    )

    return PredictionResponse(
        date=req.date,
        prediction=round(value, 2),
    )
