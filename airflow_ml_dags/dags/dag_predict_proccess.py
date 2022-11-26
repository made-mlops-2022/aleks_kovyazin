from airflow import DAG
from airflow.operators.python import PythonOperator
from packages.predicting_process import predictor
from datetime import datetime


with DAG("data_predict_dag", start_date=datetime(2022, 11, 22),
    schedule_interval='@daily', catchup=False) as dag:

        data_trainer = PythonOperator(
            task_id='data_predict',
            python_callable=predictor
        )