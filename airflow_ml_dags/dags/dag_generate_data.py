from airflow import DAG
from airflow.operators.python import PythonOperator
from packages.gen2_data import gen_data
from datetime import datetime

# ds = "{{ ds }}"

with DAG("data_generator_dag", start_date=datetime(2022, 11, 22), \
    schedule_interval='@daily', catchup=False) as dag:

        data_generation = PythonOperator(
            task_id = "data_generation",
            python_callable= gen_data
        )

# with DAG("data_generator_dag", start_date=datetime(2022, 11, 22), \
#     schedule_interval='@daily', catchup=False) as dag:

#         data_generation = DockerOperator(
#             task_id = "data_generation",
#             image = "gen_data:latest",
#             command='echo "blabla"',
#             docker_url='unix://var/run/docker.sock',
#             network_mode='bridge'
#         )

