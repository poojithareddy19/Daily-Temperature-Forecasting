import hashlib
import json

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge

from src.config import PROJECT_ROOT, load_config
from src.logger import get_logger
from src.models.baselines import baseline_predictions
from src.models.evaluate import mae, rmse
from src.models.split import temporal_train_test_split

logger = get_logger(__name__)

TARGET = "temp"
DROP_COLS = ["date", "temp"]


def load_params() -> dict:
    with open(PROJECT_ROOT / "params.yaml") as f:
        return yaml.safe_load(f)


def build_model(train_params: dict):
    name = train_params["model"]
    if name == "ridge":
        return Ridge(alpha=train_params["ridge_alpha"])
    if name == "random_forest":
        return RandomForestRegressor(
            n_estimators=train_params["n_estimators"],
            max_depth=train_params["max_depth"],
            random_state=train_params["random_state"],
            n_jobs=-1,
        )
    raise ValueError(f"Unknown model: {name}")


def feature_columns(columns, model_name: str) -> list:
    features = [c for c in columns if c not in DROP_COLS]
    if model_name == "ridge":
        # A linear model cannot use raw day-of-year (it is not linear in temperature);
        # doy_sin and doy_cos carry the season instead.
        features.remove("dayofyear")
    return features


def main() -> dict:
    cfg, params = load_config(), load_params()

    df = pd.read_csv(
        PROJECT_ROOT / cfg["data"]["processed_path"],
        parse_dates=["date"],
    )

    train_df, test_df = temporal_train_test_split(
        df,
        params["train"]["test_size"],
    )

    features = feature_columns(df.columns, params["train"]["model"])
    model = build_model(params["train"])

    mlflow.set_experiment("temperature-forecast")

    with mlflow.start_run():
        mlflow.log_params(params["train"])

        model.fit(
            train_df[features],
            train_df[TARGET],
        )

        preds = model.predict(test_df[features])

        metrics = {
            "rmse": rmse(test_df[TARGET], preds),
            "mae": mae(test_df[TARGET], preds),
        }

        mlflow.log_metrics(metrics)

        base = baseline_predictions(df, train_df, test_df)
        metrics["baselines"] = {
            name: {"rmse": rmse(test_df[TARGET], p), "mae": mae(test_df[TARGET], p)}
            for name, p in base.items()
        }
        for name, m in metrics["baselines"].items():
            mlflow.log_metric(f"baseline_{name}_rmse", m["rmse"])

        mlflow.sklearn.log_model(model, "model")

        model_path = PROJECT_ROOT / cfg["paths"]["model_path"]
        model_path.parent.mkdir(parents=True, exist_ok=True)

        joblib.dump(
            {
                "model": model,
                "features": features,
            },
            model_path,
        )
        # Lets /health prove which model file is live.
        metrics["model_sha256"] = hashlib.sha256(model_path.read_bytes()).hexdigest()
        mlflow.set_tag("model_sha256", metrics["model_sha256"])

        with open(PROJECT_ROOT / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
            f.write("\n")  # keep pre-commit's end-of-file hook and DVC's hash in agreement

        logger.info("Training done. Metrics: %s", metrics)

    return metrics


if __name__ == "__main__":
    main()
