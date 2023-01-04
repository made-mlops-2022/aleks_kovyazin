import pandas as pd
import sys
sys.path.append('..')
from packages.ft_perceptron import *
import packages.source.configurations as conf
# from loguru import logger

PATH_RAW = 'dags/packages/data/raw/ds/'
PATH_PROCESSED = 'dags/packages/data/processed/ds/'
PATH_PREICT = 'dags/packages/data/predictions/ds/'

def predictor():
    X_test, y_test = read_dataset(f'{PATH_RAW}data.csv')
    nn = NN.import_nn(f'{PATH_RAW}model.json')

    X_test = nn.scale_transform(X_test)
    layer_activations = nn.forward(X_test)
    logits = layer_activations[-1]
    probability = nn.predict_proba(logits)
    res = []
    for prob in (probability):
        res.append(np.argmax(prob))
    res = pd.DataFrame(res)
    res.to_csv('{PATH_PREICT}predictions.csv', index=False,header=False)
    
    loss = round(nn.loss(X_test, y_test).mean(), 3)
    print(loss)
    # logger.info(f'Test loss: {loss}')

if __name__=='__main__':
    predictor()