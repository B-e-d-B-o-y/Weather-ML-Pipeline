import logging
import os
import pickle
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.empty import EmptyOperator

def check_data_quality(**context):
    conf = context['dag_run'].conf or {}
    
    resp = requests.get("https://api.open-meteo.com/v1/forecast", 
                       params={'latitude': 43.12, 'longitude': 131.88,
                               'hourly': 'temperature_2m', 'forecast_days': 1})
    data = resp.json()
    df = pd.DataFrame({'temp': data['hourly']['temperature_2m']})
    
    if len(df) < 10 or df['temp'].std() < 1.0:
        logging.warning(f"Poor data quality: {len(df)} points, std={df['temp'].std():.1f}")
        return 'skip_inference'
    logging.info(f"Good data: {len(df)} points, std={df['temp'].std():.1f}")
    return 'do_inference'

def do_inference(**context):
    conf = context['dag_run'].conf or {}
    model_path = conf.get('model_path', '')
    
    if not os.path.exists(model_path):
        logging.error(f"Model not found: {model_path}")
        return
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    resp = requests.get("https://api.open-meteo.com/v1/forecast", 
                       params={'latitude': 43.12, 'longitude': 131.88,
                               'hourly': 'temperature_2m,relativehumidity_2m,precipitation',
                               'past_days': 1, 'forecast_days': 1})
    
    data = resp.json()
    hourly = data["hourly"]
    df = pd.DataFrame(hourly)
    df['time'] = pd.to_datetime(df['time'])
    df['hour'] = df['time'].dt.hour
    df['temp_lag_1'] = df['temperature_2m'].shift(1).fillna(method='bfill')
    df['temp_lag_2'] = df['temperature_2m'].shift(2).fillna(method='bfill')
    df['temp_lag_3'] = df['temperature_2m'].shift(3).fillna(method='bfill')
    df = df.dropna()
    
    features = ['temperature_2m', 'relativehumidity_2m', 'precipitation', 
                'hour', 'temp_lag_1', 'temp_lag_2', 'temp_lag_3']
    predictions = model.predict(df[features])
    
    plt.figure(figsize=(12, 6))
    plt.plot(df['time'], df['temperature_2m'], 'o-', label='Actual', linewidth=2)
    plt.plot(df['time'], predictions, 'x-', label='Predicted', linewidth=2)
    plt.title('Live Weather Inference Results')
    plt.xlabel('Time')
    plt.ylabel('Temperature (°C)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    plot_dir = '/opt/airflow/data/plots'
    os.makedirs(plot_dir, exist_ok=True)
    plot_path = os.path.join(plot_dir, f"inference_{datetime.now().strftime('%Y%m%d_%H%M')}.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    
    logging.info(f"Plot saved: {plot_path}")
    logging.info(f"Last actual: {df['temperature_2m'].iloc[-1]:.1f}°C")
    logging.info(f"Predicted next: {predictions[-1]:.1f}°C")

with DAG(
    dag_id="weather_inference_demo",
    start_date=datetime(2024, 1, 1),
    schedule_interval=None,
    catchup=False,
    tags=["inference"],
) as dag:
    
    start_inf = EmptyOperator(task_id="start_inference")
    data_check = BranchPythonOperator(task_id="data_quality_check", python_callable=check_data_quality)
    do_predict = PythonOperator(task_id="do_inference", python_callable=do_inference)
    skip_inf = EmptyOperator(task_id="skip_inference")
    end_inf = EmptyOperator(task_id="end_inference", trigger_rule="all_done")
    start_inf >> data_check >> [do_predict, skip_inf]
    [do_predict, skip_inf] >> end_inf