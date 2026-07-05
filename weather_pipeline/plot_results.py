import json
import logging
import os
from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd


def plot_results(metrics_dir: str, preds_dir: str, plots_dir: str, **context):
    os.makedirs(plots_dir, exist_ok=True)

    today_str = datetime.utcnow().strftime("%Y-%m-%d")
    metrics_path = os.path.join(metrics_dir, f"metrics_{today_str}.json")
    preds_path = os.path.join(preds_dir, f"preds_{today_str}.csv")

    if not os.path.exists(metrics_path) or not os.path.exists(preds_path):
        raise FileNotFoundError("Metrics or preds file not found for plotting")

    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    df = pd.read_csv(preds_path, parse_dates=["time"])

    plt.figure(figsize=(10, 5))
    plt.plot(df["time"], df["target_temp_next"], label="Actual temp next", marker="o")
    plt.plot(df["time"], df["predicted_temp_next"], label="Predicted temp next", marker="x")
    plt.title(
        f"Actual vs Predicted next-hour temperature\n"
        f"MAE={metrics['mae']:.3f}, RMSE={metrics['rmse']:.3f}, R2={metrics['r2']:.3f}"
    )
    plt.xlabel("Time")
    plt.ylabel("Temperature (°C)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plot_path = os.path.join(plots_dir, f"plot_{today_str}.png")
    plt.savefig(plot_path)
    plt.close()

    logging.info("Saved plot to %s", plot_path)