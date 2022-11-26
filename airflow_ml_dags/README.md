Основная часть: <br />

Поднимите airflow локально, используя docker compose (можно использовать из примера https://github.com/made-ml-in-prod-2021/airflow-examples/) <br />

1. Реализуйте dag, который генерирует данные для обучения модели (генерируйте данные -- можете использовать как генератор синтетики из первой дз, так и что-то из датасетов sklearn). Вам важно проэмулировать ситуации постоянно поступающих данных (5 баллов) ```+5``` <br />
<br />
записывайте данные в /data/raw/{{ ds }}/data.csv и /data/raw/{{ ds }}/target.csv <br />

2. Реализуйте dag, который обучает модель еженедельно, используя данные за текущий день. В вашем пайплайне должно быть как минимум 4 стадии, но дайте волю своей фантазии =) (10 баллов) ```+10``` <br />
<br />
подготовить данные для обучения (например, считать из /data/raw/{{ ds }} и положить /data/processed/{{ ds }}/train_data.csv) <br />
расплитить их на train/val <br />
обучить модель на train, сохранить в /data/models/{{ ds }} <br />
провалидировать модель на val (сохранить метрики к модельке) <br />

3. Реализуйте dag, который использует модель ежедневно (5 баллов) ```+5``` <br />
<br />
принимает на вход данные из пункта 1 (data.csv) <br />
считывает путь до модельки из airflow variables (идея в том, что когда нам нравится другая модель и мы хотим ее на прод) <br />
делает предсказание и записывает их в /data/predictions/{{ ds }}/predictions.csv <br />

4. Вы можете выбрать 2 пути для выполнения ДЗ: <br />
<br />
поставить все необходимые пакеты в образ с airflow и использовать BashOperator, PythonOperator (1 балл) ```+1``` <br />
<br />
использовать DockerOperator -- тогда выполнение каждой из тасок должно запускаться в собственном контейнере <br />
<br />
один из дагов реализован с помощью DockerOperator (5 баллов) <br />
все даги реализованы только с помощью DockerOperator (пример https://github.com/made-ml-in-prod-2021/airflow-examples/blob/main/dags/11_docker.py). По технике, вы можете использовать такую же структуру как в примере, пакуя в разные докеры скрипты, можете использовать общий докер с вашим пакетом, но с разными точками входа для разных тасок. Прикольно, если вы покажете, что для разных тасок можно использовать разный набор зависимостей (10 баллов) <br />
https://github.com/made-ml-in-prod-2021/airflow-examples/blob/main/dags/11_docker.py#L27 в этом месте пробрасывается путь с хостовой машины, используйте здесь путь типа /tmp или считывайте из переменных окружения. <br />

5. Традиционно, самооценка (1 балл) ```+1``` <br />

## Running apache airflow 2.0 in docker with local executor.
Here are the steps to take to get airflow 2.0 running with docker on your machine. 
1. Clone this repo
1. Create dags, logs and plugins folder inside the project directory
```bash
mkdir ./dags ./logs ./plugins
```
1. Install docker desktop application if you don't have docker running on your machine
- [Download Docker Desktop Application for Mac OS](https://hub.docker.com/editions/community/docker-ce-desktop-mac)

1. Launch airflow by docker-compose
```bash
docker-compose up -d
```
1. Check the running containers
```bash
docker ps
```
1. Open browser and type http://0.0.0.0:8080 to launch the airflow webserver

![](images/screenshot_airflow_docker.png)

Create Dockerfile for libs

docker build . --tag extending_airflow:latetst

change version of the airflow in docker-compose 

docker-compose up -d --no-deps --build airflow-webserver airflow-scheduler

docker compose up