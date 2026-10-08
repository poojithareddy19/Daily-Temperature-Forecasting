import pandas as pd


def baseline_predictions(
    full_df: pd.DataFrame, train_df: pd.DataFrame, test_df: pd.DataFrame
) -> dict:
    """Naive forecasts for the test rows. Each one uses only past data."""
    clim = train_df.groupby("dayofyear")["temp"].mean()
    # Same day last year: 365 rows back in the full, date-sorted frame.
    seasonal = full_df["temp"].shift(365).iloc[-len(test_df) :].reset_index(drop=True)
    return {
        "persistence": test_df["lag_1"],
        "mean_7d": test_df["roll_mean_7"],
        "climatology": test_df["dayofyear"].map(clim).fillna(train_df["temp"].mean()),
        "seasonal_naive": seasonal,
    }
