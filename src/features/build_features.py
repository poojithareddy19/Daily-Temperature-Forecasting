import pandas as pd
import yaml
from src.config import PROJECT_ROOT, load_config
from src.logger import get_logger
logger = get_logger(__name__)

def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["month"] = df["date"].dt.month
    df["dayofyear"] = df["date"].dt.dayofyear
    df["dayofweek"] = df["date"].dt.dayofweek
    return df

def add_lag_features(df: pd.DataFrame, lags, roll_windows) -> pd.DataFrame:
    df = df.copy()
    for lag in lags:
        df[f"lag_{lag}"] = df["temp"].shift(lag)
    for w in roll_windows:
        df[f"roll_mean_{w}"] = df["temp"].shift(1).rolling(w).mean()
    return df

def build_features(df: pd.DataFrame, lags, roll_windows) -> pd.DataFrame:
    df = df.sort_values("date").reset_index(drop=True)
    df = add_calendar_features(df)
    df = add_lag_features(df, lags, roll_windows)
    return df.dropna().reset_index(drop=True)

def main() -> None:
    cfg = load_config()
    with open(PROJECT_ROOT / "params.yaml") as f:
        params = yaml.safe_load(f)
    raw = pd.read_csv(PROJECT_ROOT / cfg["data"]["raw_path"])
    raw.columns = ["date", "temp"]
    raw["date"] = pd.to_datetime(raw["date"])
    raw["temp"] = pd.to_numeric(raw["temp"], errors="coerce")
    raw = raw.dropna().reset_index(drop=True)
    feats = build_features(raw, params["features"]["lags"], params["features"]["roll_windows"])
    out = PROJECT_ROOT / cfg["data"]["processed_path"]
    out.parent.mkdir(parents=True, exist_ok=True)
    feats.to_csv(out, index=False)
    logger.info("Wrote %d rows, %d cols to %s", len(feats), feats.shape[1], out)
if __name__ == "__main__":
    main()