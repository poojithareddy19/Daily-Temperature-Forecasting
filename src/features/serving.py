from datetime import date as Date
from datetime import timedelta

import pandas as pd

from src.features.build_features import add_calendar_features, add_lag_features


def features_from_history(date: Date, recent_temps: list, lags, roll_windows) -> dict:
    """Features for `date` from the readings before it. recent_temps[0] is yesterday.

    Rebuilds the history as a dated series and runs the same functions used in training,
    so serving cannot drift away from build_features.
    """
    n = max(max(lags), max(roll_windows))
    if len(recent_temps) < n:
        raise ValueError(f"need at least {n} recent temperatures, got {len(recent_temps)}")

    history = list(reversed(recent_temps[:n]))
    dates = [pd.Timestamp(date) - timedelta(days=n - i) for i in range(n + 1)]
    df = pd.DataFrame({"date": dates, "temp": history + [float("nan")]})

    df = add_calendar_features(df)
    df = add_lag_features(df, lags, roll_windows)
    return df.iloc[-1].drop(["date", "temp"]).to_dict()
