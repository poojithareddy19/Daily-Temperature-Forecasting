import json

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestRegressor

from src.config import PROJECT_ROOT, load_config
from src.logger import get_logger
from src.models.evaluate import mae, rmse
from src.models.split import temporal_train_test_split

logger = get_logger(__name__)

TARGET = "temp"
DROP_COLS = ["date", "temp"]


def load_params() -> dict:
    with open(PROJECT_ROOT / "params.yaml") as f:
        return yaml.safe_load(f)


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

    features = [c for c in df.columns if c not in DROP_COLS]

    model = RandomForestRegressor(
        n_estimators=params["train"]["n_estimators"],
        max_depth=params["train"]["max_depth"],
        random_state=params["train"]["random_state"],
        n_jobs=-1,
    )

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

        with open(PROJECT_ROOT / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

        logger.info("Training done. Metrics: %s", metrics)

    return metrics


if __name__ == "__main__":
    main()
