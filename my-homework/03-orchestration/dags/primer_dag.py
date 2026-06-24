from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime

def saludar():
    print("Hola desde Airflow! Este es mi primer DAG.")

with DAG(
    dag_id="primer_dag",
    start_date=datetime(2026, 1, 1),
    schedule="@daily",
    catchup=False
) as dag:
    tarea_saludo = PythonOperator(
        task_id="saludar",
        python_callable=saludar
    )
