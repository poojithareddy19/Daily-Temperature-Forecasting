import pandas as pd

from src.features.build_features import build_features


def test_build_features_creates_expected_columns():
    df = pd.DataFrame({
        "date": pd.date_range("2020-01-01", periods=60, freq="D"),
        "temp": range(60),
    })

    out = build_features(
        df,
        lags=[1, 2, 3, 7, 14],
        roll_windows=[7, 30],
    )

    for col in [
        "lag_1",
        "lag_14",
        "roll_mean_7",
        "roll_mean_30",
        "month",
        "dayofweek",
    ]:
        assert col in out.columns

    assert out["lag_14"].isna().sum() == 0
    assert len(out) < len(df)