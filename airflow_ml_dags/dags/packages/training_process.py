from sklearn.model_selection import train_test_split
import pandas as pd
import sys
sys.path.append('..')
from packages.ft_perceptron import *
import packages.source.configurations as conf
# from loguru import logger
import pickle

PATH_RAW = 'dags/packages/data/raw/ds/'
PATH_PROCESSED = 'dags/packages/data/processed/ds/'

def prepare_data():
    df = pd.read_csv(f'{PATH_RAW}data.csv')
    df.columns = list(df.iloc[0,:])
    x = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    feature_size = x.shape[1]
    nn = create_network(feature_size)
    data = nn.scale_fit_transform(x)
    targets = y
    with open(f'{PATH_RAW}data.pkl', 'wb') as file:
        pickle.dump(data, file)
    with open(f'{PATH_RAW}targets.pkl', 'wb') as file:
        pickle.dump(targets, file)

def splitter():
    with open(f'{PATH_RAW}data.pkl', 'rb') as file:
        X = pickle.load(file)
    with open(f'{PATH_RAW}targets.pkl', 'rb') as file:
        y = pickle.load(file)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, train_size=conf.VALID_SIZE, stratify=y)
    pd.DataFrame(X_train).to_csv(f"{PATH_PROCESSED}data_train.csv",header=False, index=False)
    pd.DataFrame(y_train).to_csv(f"{PATH_PROCESSED}target_train.csv",header=False, index=False)
    pd.DataFrame(X_val).to_csv(f"{PATH_PROCESSED}data_val.csv",header=False, index=False)
    pd.DataFrame(y_val).to_csv(f"{PATH_PROCESSED}target_val.csv",header=False, index=False)
    

def trainer():
    X_train = pd.read_csv(f"{PATH_PROCESSED}data_train.csv").values
    y_train = pd.read_csv(f"{PATH_PROCESSED}target_train.csv").values
    X_val = pd.read_csv(f"{PATH_PROCESSED}data_val.csv").values
    y_val = pd.read_csv(f"{PATH_PROCESSED}target_val.csv").values
    print(len(X_train[0]))
    nn = create_network(len(X_train[0]))
    val_log = []
    train_log = []
    for epoch in range(conf.EPOCHS):
        loss = 0
        steps = 0
        for X_batch, y_batch in iterate_batch(X_train, y_train, batch_size=conf.BATCH_SIZE):
            loss += nn.train(X_batch, y_batch)
            steps += 1
        val_log.append(nn.loss(X_val, y_val))
        train_log.append(loss / steps)
        # logger.info(
        #     f"Epoch {epoch + 1}/{conf.EPOCHS} --- loss: {round(train_log[-1], 3)} --- val_loss "
        #     f"{round(val_log[-1], 3)}"
        # )
        nn.memorize_best_weights(val_log[-1], epoch)
    pd.DataFrame({'val_log':val_log, 'train_log':train_log}).to_csv(f"{PATH_PROCESSED}val_loss.csv",index=False)
    nn.use_best_weights()
    nn.export_nn(f'{PATH_RAW}model.json')


def plotter():
    df_plot = pd.read_csv(f"{PATH_PROCESSED}val_loss.csv").dropna()
    val_losses = df_plot.val_log.tolist()
    train_losses = df_plot.train_log.tolist()
    val_losses = list(map(float, val_losses))
    train_losses = list(map(float, train_losses))
    for val_i, train_i in zip(range(len(val_losses)), range(len(train_losses))):
        plotting(val_losses[:val_i], train_losses[:train_i])
        

if __name__=='__main__':
    print(trainer())