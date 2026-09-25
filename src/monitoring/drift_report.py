import pandas as pd
from evidently import ColumnMapping
from evidently.metric_preset import DataDriftPreset
from evidently.report import Report

from src.config import PROJECT_ROOT, load_config
from src.logger import get_logger

logger = get_logger(__name__)

REFERENCE_SHARE = 0.7  # first 70% of the series is the reference window


def main() -> None:
    cfg = load_config()
    df = pd.read_csv(PROJECT_ROOT / cfg["data"]["processed_path"], parse_dates=["date"])

    split = int(len(df) * REFERENCE_SHARE)
    reference, current = df.iloc[:split], df.iloc[split:]
    logger.info("Reference window: %d rows, current window: %d rows", len(reference), len(current))

    # Declare `date` as the datetime axis so it is not scored as a feature.
    mapping = ColumnMapping(datetime="date")
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference, current_data=current, column_mapping=mapping)

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
    main()
