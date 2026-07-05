import json
import logging
import os
import pickle  # ← Встроенный, joblib не нужен
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def train_model(processed_dir: str, metrics_dir: str, preds_dir: str, models_dir: str, **context):
    os.makedirs(metrics_dir, exist_ok=True)
    os.makedirs(preds_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    today_str = datetime.utcnow().strftime("%Y-%m-%d")
    train_path = os.path.join(processed_dir, f"weather_train_{today_str}.csv")
    test_path = os.path.join(processed_dir, f"weather_test_{today_str}.csv")

    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError(f"Train or test file not found: {train_path}, {test_path}")

    train_df = pd.read_csv(train_path, parse_dates=["time"])
    test_df = pd.read_csv(test_path, parse_dates=["time"])
    logging.info("Train shape=%s, Test shape=%s", train_df.shape, test_df.shape)

    feature_cols = ["temperature_2m", "relativehumidity_2m", "precipitation", "hour", "temp_lag_1", "temp_lag_2", "temp_lag_3"]
    target_col = "target_temp_next"

    X_train = train_df[feature_cols]
    y_train = train_df[target_col]
    X_test = test_df[feature_cols]
    y_test = test_df[target_col]

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    metrics = {
        "mae": float(mae),
        "mse": float(mse),
        "rmse": float(rmse),
        "r2": float(r2),
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
    }

    logging.info("Metrics: MAE=%.3f, RMSE=%.3f, R2=%.3f", mae, rmse, r2)

    # Сохранение метрик
    metrics_path = os.path.join(metrics_dir, f"metrics_{today_str}.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    # Сохранение предсказаний
    preds_df = test_df[["time", target_col]].copy()
    preds_df["predicted_temp_next"] = y_pred
    preds_path = os.path.join(preds_dir, f"preds_{today_str}.csv")
    preds_df.to_csv(preds_path, index=False)

    # Сохранение модели с pickle
    model_path = os.path.join(models_dir, f"weather_model_{today_str}.pkl")
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)

    logging.info("Saved metrics: %s", metrics_path)
    logging.info("Saved predictions: %s", preds_path)
    logging.info("Saved model: %s", model_path)

    # ✅ Возвращаем метрики для BranchOperator (XCom)
    return metrics
