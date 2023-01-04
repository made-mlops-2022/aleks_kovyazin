import pandas as pd
from loguru import logger
import sys, re
from pathlib import Path
sys.path.append(f'{Path(__file__).resolve(strict=True).parent.parent}')
sys.path.append('../..')
from ml_project.source.net import NN
import ml_project.source.configurations as conf
from ml_project.ft_perceptron import predict,train 

BASE_DIR = Path(__file__).resolve(strict=True).parent
# print(Path(__file__).resolve(strict=True).parent.parent.parent)


def predict_pipeline(text):
    text = re.sub(r'[!@#$(),\n"%^*?\:;~`0-9]', " ", text)
    text = re.sub(r"[[]]", " ", text)
    text = text.lower()
    return predict(text)

def train_pipeline(text):
    text = re.sub(r'[!@#$(),\n"%^*?\:;~`0-9]', " ", text)
    text = re.sub(r"[[]]", " ", text)
    text = text.lower()
    return train(text)



