import pandas as pd

from src.models.evaluate import mae, rmse
from src.models.split import temporal_train_test_split


def test_metrics_zero_for_perfect_prediction():
    y = [1.0, 2.0, 3.0]

    assert rmse(y, y) == 0.0
    assert mae(y, y) == 0.0


def test_rmse_known_value():
    assert rmse([0, 0], [1, 1]) == 1.0


def test_temporal_split_preserves_order():
    df = pd.DataFrame({"x": range(100)})

    train, test = temporal_train_test_split(
        df,
        test_size=0.2,
    )

    assert len(test) == 20
    assert train["x"].iloc[-1] < test["x"].iloc[0]


def test_baselines_one_prediction_per_test_row():
    from src.models.baselines import baseline_predictions

    n = 800
    df = pd.DataFrame(
        {
            "temp": [float(i % 50) for i in range(n)],
            "dayofyear": [i % 365 + 1 for i in range(n)],
        }
    )
    df["lag_1"] = df["temp"].shift(1)
    df["roll_mean_7"] = df["temp"].shift(1).rolling(7).mean()
    df = df.dropna().reset_index(drop=True)

    train, test = temporal_train_test_split(df, test_size=0.2)
    preds = baseline_predictions(df, train, test)

    assert set(preds) == {"persistence", "mean_7d", "climatology", "seasonal_naive"}
    for name, p in preds.items():
        assert len(p) == len(test), name
        assert p.isna().sum() == 0, name
