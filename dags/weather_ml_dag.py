import sys
import os
sys.path.insert(0, '/opt/airflow')

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from airflow.operators.empty import EmptyOperator

from weather_pipeline.fetch_weather import fetch_weather
from weather_pipeline.preprocess import preprocess_data
from weather_pipeline.train_model import train_model

default_args = {
    "owner": "airflow",
    "retries": 3,
    "retry_delay": timedelta(minutes=2),
}

with DAG(
    dag_id="weather_ml_pipeline_enhanced",
    default_args=default_args,
    start_date=datetime(2024, 1, 1),
    schedule_interval="@daily",
    catchup=False,
    description="Training pipeline: data → preprocess → train → trigger inference",
    tags=["training", "ml"],
) as dag:

    start = EmptyOperator(task_id="start")

    fetch_weather_task = PythonOperator(
        task_id="fetch_weather",
        python_callable=fetch_weather,
        op_kwargs={
            "data_dir": "/opt/airflow/data/raw",
            "lat": 43.12,
            "lon": 131.88,
            "past_days": 7,
            "forecast_days": 1,
        },
    )

    preprocess_task = PythonOperator(
        task_id="preprocess_data",
        python_callable=preprocess_data,
        op_kwargs={
            "raw_dir": "/opt/airflow/data/raw",
            "processed_dir": "/opt/airflow/data/processed",
        },
    )

    train_model_task = PythonOperator(
        task_id="train_model",
        python_callable=train_model,
        op_kwargs={
            "processed_dir": "/opt/airflow/data/processed",
            "metrics_dir": "/opt/airflow/data/metrics",
            "preds_dir": "/opt/airflow/data/preds",
            "models_dir": "/opt/airflow/data/models",
        },
    )

    trigger_inference = TriggerDagRunOperator(
        task_id="trigger_inference",
        trigger_dag_id="weather_inference_demo",
        conf={"model_path": "/opt/airflow/data/models/weather_model_{{ ds }}.pkl"},
    )

    end = EmptyOperator(task_id="end")

    start >> fetch_weather_task >> preprocess_task >> train_model_task >> trigger_inference >> end