import pandas as pd
import pytest

from src.data.validate import DataValidationError, validate


def make_raw(temps, dates=None):
    """Raw rows as build_features.main() sees them: parsed dates, temps coerced to numbers."""
    dates = dates or pd.date_range("1990-01-01", periods=len(temps), freq="D")
    df = pd.DataFrame({"date": pd.to_datetime(dates), "temp": temps})
    df["temp"] = pd.to_numeric(df["temp"], errors="coerce")
    return df


def test_valid_data_passes():
    validate(make_raw([10.0, 12.5, -3.0, 25.1]))


def test_missing_column_is_rejected():
    df = make_raw([10.0, 11.0]).drop(columns="temp")

    with pytest.raises(DataValidationError, match="Missing columns"):
        validate(df)


def test_missing_temperature_is_rejected():
    with pytest.raises(DataValidationError, match="missing temperature"):
        validate(make_raw([10.0, None, 12.0]))


def test_non_numeric_temperature_is_rejected_after_coercion():
    # The upstream file once had readings like "?0.2"; coercion turns them into NaN.
    with pytest.raises(DataValidationError, match="missing temperature"):
        validate(make_raw(["10.0", "?0.2", "12.0"]))


@pytest.mark.parametrize("bad", [99.0, -51.0])
def test_out_of_range_temperature_is_rejected(bad):
    with pytest.raises(DataValidationError, match="plausible range"):
        validate(make_raw([10.0, bad, 12.0]))


def test_unsorted_dates_are_rejected():
    dates = ["1990-01-02", "1990-01-01", "1990-01-03"]

    with pytest.raises(DataValidationError, match="not sorted"):
        validate(make_raw([10.0, 11.0, 12.0], dates))
