from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
from pipelines.training_pipeline import run_training_pipeline

default_args = {
    'owner': 'airflow',
    'start_date': datetime(2024, 8, 27),
    'retries': 1,
}

dag = DAG(
    'training_dag',
    default_args=default_args,
    description='A simple training DAG',
    schedule_interval='@once',
)

train_task = PythonOperator(
    task_id='run_training_pipeline',
    python_callable=run_training_pipeline,
    dag=dag,
)
