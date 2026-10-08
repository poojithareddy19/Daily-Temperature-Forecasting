"""Training and serving must compute identical features for the same day."""

from pathlib import Path

import pandas as pd
import pytest

from src.features.build_features import build_features
from src.features.serving import features_from_history

LAGS = [1, 2, 3, 7, 14]
ROLL_WINDOWS = [7, 30]
FIXTURE = Path(__file__).parent / "fixtures" / "melbourne_60_days.csv"


def test_serving_features_match_training_features():
    raw = pd.read_csv(FIXTURE)
    raw.columns = ["date", "temp"]
    raw["date"] = pd.to_datetime(raw["date"])

    train_row = build_features(raw, LAGS, ROLL_WINDOWS).iloc[-1]
    recent = raw["temp"].iloc[-31:-1].tolist()[::-1]  # 30 days before the last one, newest first

    served = features_from_history(raw["date"].iloc[-1], recent, LAGS, ROLL_WINDOWS)

    expected = train_row.drop(["date", "temp"])
    assert set(served) == set(expected.index)
    for col, value in expected.items():
        assert served[col] == pytest.approx(value, abs=1e-9), col


def test_too_little_history_is_rejected():
    with pytest.raises(ValueError):
        features_from_history(pd.Timestamp("1991-01-01"), [10.0] * 29, LAGS, ROLL_WINDOWS)
