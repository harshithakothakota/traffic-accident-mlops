"""Airflow DAG: runs the ETL steps in etl.py in order, once a day."""
import sys
from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.operators.python import PythonOperator

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # project folder
import etl  # noqa: E402

with DAG(
    dag_id="accident_etl_pipeline",
    start_date=datetime(2026, 1, 1),
    schedule="@daily",
    catchup=False,
    default_args={"retries": 1, "retry_delay": timedelta(minutes=2)},
    tags=["traffic-accidents"],
) as dag:
    tasks = {step.__name__: PythonOperator(task_id=step.__name__, python_callable=step)
             for step in etl.STEPS}

    t = tasks
    t["extract_accident_data"] >> t["extract_weather_data"]
    t["extract_accident_data"] >> [t["stage_collisions"], t["stage_vehicles"],
                                   t["stage_casualties"]]
    [t["stage_collisions"], t["stage_vehicles"], t["stage_casualties"]] >> t["validate_data"]
    [t["validate_data"], t["extract_weather_data"]] >> t["join_accident_weather"]
    t["join_accident_weather"] >> t["load_postgresql"] >> t["build_analytical_mart"]
