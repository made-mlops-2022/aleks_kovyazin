Весь код должен находиться в том же репозитории, но в отдельной папке online_inference. <br />

Основная часть: <br />

1. Оберните inference вашей модели в rest сервис на FastAPI, должен быть endpoint /predict (3 балла) 
После сборки Dockerfile и запуска команды docker run -p 80:80 ft_perceptron-app заходим на localhost:80/ localhost:80/train localhost:80/predict
```+3```<br />

2. Напишите endpoint /health, который должен возращать 200, если ваша модель готова к работе (такой чек особенно актуален, если делаете доп задание про скачивание из хранилища) (1 балл) ```+1```<br />
После сборки Dockerfile и запуска команды docker run -p 80:80 ft_perceptron-app заходим на localhost:80<br />
3. Напишите unit тест для /predict (https://fastapi.tiangolo.com/tutorial/testing/, https://flask.palletsprojects.com/en/1.1.x/testing/) (3 балла) ```+3```<br />
Тесты написаны в файле app/test_request.py <br />

4. Напишите скрипт, который будет делать запросы к вашему сервису (2 балла) ```+2```<br />
Написан скрипт на bash app/curl_requests.sh <br />

5. Напишите Dockerfile, соберите на его основе образ и запустите локально контейнер (docker build, docker run). Внутри контейнера должен запускаться сервис, написанный в предущем пункте. Закоммитьте его, напишите в README.md корректную команду сборки (4 балла) ```+4```<br />
Написан и описан. Запускается внутри контейнера после входа <br />
docker exec -it id_container bash<br />
bash ./curl_requests.sh <br />


6. Опубликуйте образ в https://hub.docker.com/, используя docker push (вам потребуется зарегистрироваться) (2 балла) ```+2```<br />
Опубдикован https://hub.docker.com/r/krastykovyaz/ft_perceptron-app <br />


7. Опишите в README.md корректные команды docker pull/run, которые должны привести к тому, что локально поднимется на inference ваша модель. Убедитесь, что вы можете протыкать его скриптом из пункта 3 (1 балл) ```+1```<br />
docker login <br />
docker tag d1c86cfc3fd6 krastykovyaz/ft_perceptron-app:0.0.1 <br />
docker push krastykovyaz/ft_perceptron:0.0.1 <br />

8. Проведите самооценку - распишите в реквесте какие пункты выполнили и на сколько баллов, укажите общую сумму баллов (1 балл) ```+1```<br />
Самооценка проведедена, итоговый бал внизу.<br />

Дополнительная часть: <br />
<br />
9. Ваш сервис скачивает модель из S3 или любого другого хранилища при старте, путь для скачивания передается через переменные окружения (+2 доп балла) ```+2```<br />
Сервис скачивает и распаковывает датасет с kuggle.com. В dockerfile прописаны все команды. <br />
10. ~~Оптимизируйте размер docker image. Опишите в README.md, что вы предприняли для сокращения размера и каких результатов удалось добиться. Должно получиться мини исследование -- я сделал тото и получился такой-то результат (+2 доп балла) https://docs.docker.com/develop/develop-images/dockerfile_best-practices/~~ <br />
11. ~~Сделайте валидацию входных данных https://pydantic-docs.helpmanual.io/usage/validators/ . Например, порядок колонок не совпадает с трейном, типы, допустимые максимальные и минимальные значения. Проявите фантазию, это доп. баллы, проверка не должна быть тривиальной. Вы можете сохранить вместе с моделью доп информацию о структуре входных данных, если это нужно (+2 доп балла). https://fastapi.tiangolo.com/tutorial/handling-errors/ -- возращайте 400, в случае, если валидация не пройдена~~ <br />
<br />
<br />
Итоговое количество баллов: ```19```<br />
<br />
<br />
<br />
Hint:
Пушим в DockerHub <br />
docker build -t ft_perceptron-app . <br />
docker run -p 80:80 ft_perceptron-app <br />
docker login <br />
docker tag d1c86cfc3fd6 krastykovyaz/ft_perceptron-app:0.0.1 <br />
docker push krastykovyaz/ft_perceptron:0.0.1 <br />
Для проверки путей внутри контейнера
docker exec -it id_container bash<br />
