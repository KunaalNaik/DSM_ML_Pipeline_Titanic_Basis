from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
from pipelines.inference_pipeline import run_inference_pipeline

default_args = {
    'owner': 'airflow',
    'start_date': datetime(2024, 8, 27),
    'retries': 1,
}

dag = DAG(
    'inference_dag',
    default_args=default_args,
    description='A simple inference DAG',
    schedule_interval='@once',
)

inference_task = PythonOperator(
    task_id='run_inference_pipeline',
    python_callable=run_inference_pipeline,
    dag=dag,
)
