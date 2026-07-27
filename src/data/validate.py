import pandas as pd
from src.config import PROJECT_ROOT, load_config
from src.logger import get_logger
logger = get_logger(__name__)
EXPECTED_COLUMNS = {"date", "temp"}
def validate(df: pd.DataFrame) -> None:
    missing = EXPECTED_COLUMNS - set(df.columns)
    assert not missing, f"Missing columns: {missing}"
    assert df["temp"].notna().all(), "Found missing temperature values"
    assert df["temp"].between(-50, 60).all(), "Temperature outside plausible range"
    assert df["date"].is_monotonic_increasing, "Dates are not sorted ascending"
    logger.info(
        "Validation passed: %d rows, range %.1f..%.1f",
        len(df), df["temp"].min(), df["temp"].max(),
    )
if __name__ == "__main__":
    cfg = load_config()
    df = pd.read_csv(PROJECT_ROOT / cfg["data"]["raw_path"], parse_dates=["date"])
    df["temp"] = pd.to_numeric(df["temp"], errors="coerce")
    df = df.dropna().sort_values("date").reset_index(drop=True)
    validate(df)