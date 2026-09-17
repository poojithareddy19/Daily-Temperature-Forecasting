from datetime import datetime

import joblib
from fastapi import FastAPI, HTTPException

from src.api.schemas import PredictionRequest, PredictionResponse
from src.config import PROJECT_ROOT, load_config
from src.logger import get_logger


logger = get_logger(__name__)

cfg = load_config()

app = FastAPI(title="Temperature Forecast API", version="1.0.0")

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
    d = datetime.strptime(req.date, "%Y-%m-%d")

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
        raise HTTPException(status_code=503, detail="Model not loaded")

    value = float(_bundle["model"].predict([_serving_features(req)])[0])

    return PredictionResponse(date=req.date,prediction=round(value, 2))