"""Where does the production model go wrong? Groups backtest errors by calendar month.

Usage: python scripts/error_analysis.py  (run scripts/backtest.py first)
Writes reports/error_by_month.csv, reports/residuals.png and reports/rmse_by_month.png.
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import PROJECT_ROOT  # noqa: E402
from src.models.train import load_params  # noqa: E402


def main() -> None:
    model_name = load_params()["train"]["model"]
    reports = PROJECT_ROOT / "reports"
    preds = pd.read_csv(reports / "backtest_predictions.csv", parse_dates=["date"])
    preds["error"] = preds["pred"] - preds["actual"]
    preds["month"] = preds["date"].dt.month

    by_month = (
        preds.groupby(["month", "model"])["error"]
        .agg(
            rmse=lambda e: (e**2).mean() ** 0.5,
            bias="mean",
            n="size",
        )
        .reset_index()
    )
    by_month.to_csv(reports / "error_by_month.csv", index=False)

    ours = preds[preds["model"] == model_name].sort_values("date")
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.plot(ours["date"], ours["error"], linewidth=0.6)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title(f"{model_name} residuals (prediction - actual), walk-forward 1986-1990")
    ax.set_ylabel("deg C")
    fig.tight_layout()
    fig.savefig(reports / "residuals.png", dpi=120)
    plt.close(fig)

    table = by_month.pivot(index="month", columns="model", values="rmse")
    ax = table[["persistence", model_name]].plot.bar(figsize=(10, 3.5), rot=0)
    ax.set_title("RMSE by month, walk-forward 1986-1990")
    ax.set_ylabel("RMSE (deg C)")
    ax.figure.tight_layout()
    ax.figure.savefig(reports / "rmse_by_month.png", dpi=120)
    plt.close(ax.figure)

    print(table.round(3).to_string())


if __name__ == "__main__":
    main()
