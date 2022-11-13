FROM tiangolo/uvicorn-gunicorn-fastapi:python3.9

COPY ./requirements.txt /app/requirements.txt
COPY ./app/curl_requests.sh /app/curl_requests.sh
COPY . /app
COPY ./.kaggle/kaggle.json /root/.kaggle/kaggle.json

RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt
RUN kaggle datasets download cherngs/heart-disease-cleveland-uci

ENV KAGGLE_USERNAME=alexkovyaz
ENV KAGGLE_KEY=40054005

RUN unzip heart-disease-cleveland-uci
RUN mv heart_cleveland_upload.csv ./app/ml_project/data/data.csv

# RUN sh /app/curl_requests.sh

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]