"""Walk-forward backtest: for each test year Y, train on every year before Y and test on Y.

A single 80/20 split is one sample; this gives five, so the model choice does not
hinge on one easy or hard year.

Usage: python scripts/backtest.py
Writes reports/backtest.csv (one row per year per model), reports/backtest_predictions.csv,
reports/backtest.md and reports/backtest_summary.json.
"""

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import PROJECT_ROOT, load_config  # noqa: E402
from src.models.evaluate import rmse  # noqa: E402
from src.models.train import TARGET, build_model, feature_columns, load_params  # noqa: E402

TEST_YEARS = range(1986, 1991)
MODELS = ["ridge", "random_forest"]


def main() -> None:
    cfg, params = load_config(), load_params()
    df = pd.read_csv(PROJECT_ROOT / cfg["data"]["processed_path"], parse_dates=["date"])
    year = df["date"].dt.year

    rows, preds = [], []
    for y in TEST_YEARS:
        train, test = df[year < y], df[year == y]
        year_preds = {"persistence": test["lag_1"].to_numpy()}
        for name in MODELS:
            model = build_model({**params["train"], "model": name})
            features = feature_columns(df.columns, name)
            model.fit(train[features], train[TARGET])
            year_preds[name] = model.predict(test[features])

        for name, p in year_preds.items():
            rows.append({"year": y, "model": name, "rmse": rmse(test[TARGET], p)})
            preds.append(
                pd.DataFrame(
                    {"date": test["date"], "model": name, "actual": test[TARGET], "pred": p}
                )
            )
        print(f"{y}: " + ", ".join(f"{r['model']}={r['rmse']:.3f}" for r in rows[-3:]))

    out_dir = PROJECT_ROOT / "reports"
    out_dir.mkdir(exist_ok=True)
    results = pd.DataFrame(rows)
    results.to_csv(out_dir / "backtest.csv", index=False)
    pd.concat(preds).to_csv(out_dir / "backtest_predictions.csv", index=False)

    summary = results.groupby("model", sort=False)["rmse"].agg(["mean", "std"])
    with open(out_dir / "backtest_summary.json", "w") as f:
        json.dump(
            {m: {"rmse_mean": r["mean"], "rmse_std": r["std"]} for m, r in summary.iterrows()},
            f,
            indent=2,
        )
        f.write("\n")

    lines = ["| Model | RMSE per year | Mean +/- std |", "|---|---|---|"]
    for m, r in summary.iterrows():
        per_year = ", ".join(f"{v:.3f}" for v in results.loc[results["model"] == m, "rmse"])
        lines.append(f"| {m} | {per_year} | {r['mean']:.3f} +/- {r['std']:.3f} |")
    (out_dir / "backtest.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
