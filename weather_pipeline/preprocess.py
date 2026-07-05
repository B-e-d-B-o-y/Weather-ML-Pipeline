import logging
import os
from datetime import datetime

import pandas as pd
import numpy as np


def preprocess_data(raw_dir: str, processed_dir: str, **context):
    os.makedirs(processed_dir, exist_ok=True)

    today_str = datetime.utcnow().strftime("%Y-%m-%d")
    raw_path = os.path.join(raw_dir, f"weather_{today_str}.csv")
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Raw file not found: {raw_path}")

    df = pd.read_csv(raw_path, parse_dates=["time"])
    logging.info("Loaded raw data: %s, shape=%s", raw_path, df.shape)

    # Валидация: выбросы и NaN
    before = len(df)
    df = df.dropna(subset=["temperature_2m"])
    df = df[(df["temperature_2m"] > -60) & (df["temperature_2m"] < 60)]
    after = len(df)
    logging.info("Dropped %d rows during validation", before - after)

    df = df.sort_values("time").reset_index(drop=True)

    # Фичи: лаги температуры и час суток
    df["hour"] = df["time"].dt.hour
    for lag in [1, 2, 3]:
        df[f"temp_lag_{lag}"] = df["temperature_2m"].shift(lag)

    # Target: температура через 1 шаг
    df["target_temp_next"] = df["temperature_2m"].shift(-1)

    df = df.dropna().reset_index(drop=True)
    logging.info("After lag/target creation shape=%s", df.shape)

    # Train / test по времени (80/20)
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()

    train_path = os.path.join(processed_dir, f"weather_train_{today_str}.csv")
    test_path = os.path.join(processed_dir, f"weather_test_{today_str}.csv")
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    logging.info("Saved train to %s shape=%s", train_path, train_df.shape)
    logging.info("Saved test to %s shape=%s", test_path, test_df.shape)