"""Evidently drift report: training features vs the inputs the API actually received.

Reference window: the training split of the feature table.
Current window: logged /predict requests, or the test split when there are fewer than
MIN_LOGGED_REQUESTS of them (always the case in CI, and on Render after a restart).

Usage: python -m src.monitoring.drift_report [--test-period]
"""

import sys

import pandas as pd
import yaml
from evidently.metric_preset import DataDriftPreset
from evidently.report import Report

from src.config import PROJECT_ROOT, load_config
from src.logger import get_logger
from src.models.split import temporal_train_test_split

logger = get_logger(__name__)

MIN_LOGGED_REQUESTS = 100
NON_FEATURES = ["date", "temp", "ts", "prediction"]


def load_current(cfg: dict, test_df: pd.DataFrame, force_test_period: bool):
    log_path = PROJECT_ROOT / cfg["paths"]["request_log"]
    logged = pd.read_json(log_path, lines=True) if log_path.exists() else pd.DataFrame()

    if force_test_period:
        return test_df, "test period (requested)"
    if len(logged) < MIN_LOGGED_REQUESTS:
        logger.warning(
            "Only %d logged requests in %s (need %d). FALLING BACK TO THE TEST PERIOD: "
            "this report says nothing about live traffic.",
            len(logged),
            log_path,
            MIN_LOGGED_REQUESTS,
        )
        return test_df, f"test period (fallback, only {len(logged)} logged requests)"
    return logged, f"{len(logged)} logged requests"


def main(force_test_period: bool = False) -> None:
    cfg = load_config()
    with open(PROJECT_ROOT / "params.yaml") as f:
        test_size = yaml.safe_load(f)["train"]["test_size"]

    df = pd.read_csv(PROJECT_ROOT / cfg["data"]["processed_path"], parse_dates=["date"])
    reference, test_df = temporal_train_test_split(df, test_size)
    current, source = load_current(cfg, test_df, force_test_period)

    features = [c for c in reference.columns if c not in NON_FEATURES and c in current.columns]
    logger.info(
        "Reference: %d training rows. Current: %s. Features: %s", len(reference), source, features
    )

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference[features], current_data=current[features])

    out = PROJECT_ROOT / "reports" / "drift.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    report.save_html(str(out))

    summary = report.as_dict()["metrics"][0]["result"]
    logger.info(
        "Drift detected in %s of %s columns (dataset drift: %s)",
        summary.get("number_of_drifted_columns"),
        summary.get("number_of_columns"),
        summary.get("dataset_drift"),
    )
    logger.info("Drift report written to %s", out)


if __name__ == "__main__":
    main(force_test_period="--test-period" in sys.argv)
