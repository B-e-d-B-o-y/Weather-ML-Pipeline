import logging
import os
from datetime import datetime

import pandas as pd
import requests

BASE_URL = "https://api.open-meteo.com/v1/forecast"


def fetch_weather(data_dir: str, lat: float, lon: float, past_days: int = 2, forecast_days: int = 1, **context):
    os.makedirs(data_dir, exist_ok=True)

    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "temperature_2m,relativehumidity_2m,precipitation",
        "past_days": past_days,
        "forecast_days": forecast_days,
        "timezone": "auto",
    }

    logging.info("Requesting weather data with params: %s", params)
    resp = requests.get(BASE_URL, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    hourly = data.get("hourly", {})
    times = hourly.get("time", [])
    temps = hourly.get("temperature_2m", [])
    humidity = hourly.get("relativehumidity_2m", [])
    precip = hourly.get("precipitation", [])

    df = pd.DataFrame(
        {
            "time": pd.to_datetime(times),
            "temperature_2m": temps,
            "relativehumidity_2m": humidity,
            "precipitation": precip,
        }
    )

    logging.info("Fetched %d rows of weather data", len(df))

    today_str = datetime.utcnow().strftime("%Y-%m-%d")
    out_path = os.path.join(data_dir, f"weather_{today_str}.csv")
    df.to_csv(out_path, index=False)
    logging.info("Saved raw weather data to %s", out_path)