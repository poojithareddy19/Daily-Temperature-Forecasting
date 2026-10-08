import pandas as pd

from src.features.build_features import add_calendar_features, build_features


def test_build_features_creates_expected_columns():
    df = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=60, freq="D"),
            "temp": range(60),
        }
    )

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
        "dayofyear",
        "doy_sin",
        "doy_cos",
    ]:
        assert col in out.columns

    assert out["lag_14"].isna().sum() == 0
    assert len(out) < len(df)


def test_cyclic_day_of_year_wraps_around_new_year():
    df = pd.DataFrame({"date": pd.to_datetime(["2020-12-31", "2021-01-01", "2021-07-01"])})

    out = add_calendar_features(df)
    pos = out[["doy_sin", "doy_cos"]].to_numpy()

    def dist(a, b):
        return float(((pos[a] - pos[b]) ** 2).sum() ** 0.5)

    # New Year's Eve and New Year's Day are neighbours; midwinter is far from both.
    assert dist(0, 1) < 0.05
    assert dist(0, 2) > 1.5
