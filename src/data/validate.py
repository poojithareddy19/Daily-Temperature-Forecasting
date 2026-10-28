import pandas as pd

from src.config import PROJECT_ROOT, load_config
from src.logger import get_logger

logger = get_logger(__name__)
EXPECTED_COLUMNS = {"date", "temp"}
# Plausible daily minimum in deg C; the API reuses these bounds for /predict input.
TEMP_MIN = -50.0
TEMP_MAX = 60.0


class DataValidationError(ValueError):
    """Raised when the raw data breaks a rule. Not an assert, so python -O keeps it."""


def validate(df: pd.DataFrame) -> None:
    missing = EXPECTED_COLUMNS - set(df.columns)
    if missing:
        raise DataValidationError(f"Missing columns: {missing}")
    if not df["temp"].notna().all():
        raise DataValidationError("Found missing temperature values")
    if not df["temp"].between(TEMP_MIN, TEMP_MAX).all():
        raise DataValidationError("Temperature outside plausible range")
    if not df["date"].is_monotonic_increasing:
        raise DataValidationError("Dates are not sorted ascending")
    logger.info(
        "Validation passed: %d rows, range %.1f..%.1f",
        len(df),
        df["temp"].min(),
        df["temp"].max(),
    )


if __name__ == "__main__":
    cfg = load_config()
    df = pd.read_csv(PROJECT_ROOT / cfg["data"]["raw_path"], parse_dates=["date"])
    # Coerced, not dropped: an unparseable reading must fail validation, as in prepare.
    df["temp"] = pd.to_numeric(df["temp"], errors="coerce")
    validate(df)
