from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
from packages.training_process import prepare_data, splitter, trainer, plotter


with DAG("data_preparator_dag", start_date=datetime(2022, 11, 22),
schedule_interval='@weekly', catchup=False) as dag:

    data_preparator = PythonOperator(
        task_id='data_prepare',
        python_callable=prepare_data
    )


    data_splitter = PythonOperator(
        task_id='data_split',
        python_callable=splitter
    )


    data_trainer = PythonOperator(
        task_id='data_train',
        python_callable=trainer
    )

    data_plotter = PythonOperator(
        task_id='data_plot',
        python_callable=plotter
    )

    data_preparator >> data_splitter >> data_trainer >> data_plotter
