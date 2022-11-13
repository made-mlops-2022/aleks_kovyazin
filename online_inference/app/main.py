from fastapi import FastAPI
from pydantic import  BaseModel
import sys
sys.path.append('..')
from app.model.model import predict_pipeline, train_pipeline

app = FastAPI()

class TextIn(BaseModel):
    text: str


class Prediction(BaseModel):
    result: str

class Train(BaseModel):
    result: str

@app.get('/')
def home():
    return {"health check": "OK"}

@app.get('/health')
def home():
    return {"health check": "200"}


@app.post("/predict", response_model=Prediction)
def predict(payload: TextIn):
    result = predict_pipeline(payload.text)
    return {'result': result}

@app.post("/train", response_model=Train)
def train(payload: TextIn):
    result = train_pipeline(payload.text)
    return {'result': result}