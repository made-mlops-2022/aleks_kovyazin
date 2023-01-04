import pandas as pd
import sys
from pathlib import Path
sys.path.append(f'{Path(__file__).resolve(strict=True).parent}')
from source.net import NN, Linear, Relu
from sklearn.model_selection import train_test_split
import numpy as np
import argparse, sys
from matplotlib import pyplot as plt
from loguru import logger
import source.configurations as conf



def read_dataset(path):
    df = pd.read_csv(path)
    df.columns = list(df.iloc[0,:])
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    return X, y



def create_network(features_size):
    network = NN()
    network.append(Linear(features_size, 200))
    network.append(Relu())
    network.append(Linear(200, 200))
    network.append(Relu())
    network.append(Linear(200, 200))
    network.append(Relu())
    network.append(Linear(200, 2))
    return network

def iterate_batch(x_, y_, batch_size):
    indices = np.random.permutation(len(y_))
    for start_idx in range(0, len(y_) - batch_size + 1, batch_size):
        slice = indices[start_idx:start_idx + batch_size]
        yield x_[slice], y_[slice]

def plotting(train_losses, val_losses, is_end=False, best_loss=None, best_epoch=None):
    if not conf.IS_PLOT:
        return
    plt.clf()
    if is_end:
        assert best_loss is not None and best_epoch is not None, \
            'Pass the best loss and best epoch to the function'
        first_step = next(i for i, val_loss in enumerate(val_losses) \
                          if val_loss < np.median(val_losses))
        plt.plot([best_epoch], [best_loss], marker='o', markersize=5, color="red")
        plt.xlim(first_step, conf.EPOCHS)
    plt.plot(val_losses, color='g', label='val loss')
    plt.plot(train_losses, ls='--', color='blue', label='train loss', alpha=0.5)

    plt.legend()
    plt.draw()
    plt.pause(conf.EPS)
    if is_end:
        plt.show()

def train(text):
    try:
        logger.info(conf.TRAIN_FILENAME)
        X, y = read_dataset(conf.TRAIN_FILENAME)
    except Exception as e:
        logger.critical(f'Error read train dataset ({repr(e)})')
        sys.exit()

    nn = create_network(X.shape[1])
    X_train, X_val, y_train, y_val = train_test_split(nn.scale_fit_transform(X), y, test_size=conf.VALID_SIZE,
                                                      stratify=y)
    val_log = []
    train_log = []
    try:
        for epoch in range(conf.EPOCHS):
            loss = 0
            steps = 0
            for X_batch, y_batch in iterate_batch(X_train, y_train, batch_size=conf.BATCH_SIZE):
                loss += nn.train(X_batch, y_batch)
                steps += 1
            val_log.append(nn.loss(X_val, y_val))
            train_log.append(loss / steps)
            logger.info(
                f"Epoch {epoch + 1}/{conf.EPOCHS} --- loss: {round(train_log[-1], 3)} --- val_loss "
                f"{round(val_log[-1], 3)}"
            )
            plotting(train_log, val_log)
            nn.memorize_best_weights(val_log[-1], epoch)
    except KeyboardInterrupt:
        logger.warning('Train is terminated !')
    else:
        plotting(train_log, val_log, is_end=True, best_epoch=nn.best_epoch, best_loss=nn.best_loss)
    nn.use_best_weights()
    nn.export_nn(conf.MODEL_DATA_PATH)
    return "Model is done !"

def predict(text):
    try:
        X_test, y_test = read_dataset(conf.TEST_FILENAME)
    except Exception as e:
        logger.critical(f'Error reading test dataset ({repr(e)})')
        sys.exit()
    try:
        nn = NN.import_nn(conf.MODEL_DATA_PATH)
    except Exception as e:
        logger.critical(f'Error import model_config ({repr(e)})')
        sys.exit()

    X_test = nn.scale_transform(X_test)
    logger.info(f'Test loss: {round(nn.loss(X_test, y_test).mean(), 3)}')
    return f'{round(nn.loss(X_test, y_test).mean(), 3)}'

def main():


    logger.remove()
    logger.add(sys.stderr, diagnose=False, backtrace=False)
    logger.add(conf.FILE_LOG_PATH, diagnose=False, backtrace=False)

    parser = argparse.ArgumentParser(description="Perceptron")

    parser.add_argument('-t', '--train', dest='train', action='store_true',
                        help='train mode')
    parser.add_argument('-p', '--predict', dest='predict', action='store_true',
                        help='predict mode')
    parser.add_argument('--plot', dest='plot', action='store_true', default=conf.IS_PLOT,
                        help='visualization')
    parser.add_argument('--valid-size', type=float, default=conf.VALID_SIZE, dest='valid_size',
                        help='split validation size (0-1)')
    parser.add_argument('-lr', '--learning-rate', type=float, default=conf.LEARNING_RATE, dest='learning_rate')
    parser.add_argument('-e', '--epochs', type=int, default=conf.EPOCHS, dest='epochs', help='number epochs')
    parser.add_argument('--batch-size', type=int, default=conf.BATCH_SIZE, dest='batch_size')
    parser.add_argument('--train-path', '--train-filename', default=conf.TRAIN_FILENAME,
                        help='train dataset filename', dest='train_path')
    parser.add_argument('--test-path', '--test-filename', default=conf.TEST_FILENAME, help='test dataset filename',
                        dest='test_path')
    args = parser.parse_args()

    conf.VALID_SIZE = args.valid_size
    conf.EPOCHS = args.epochs
    conf.BATCH_SIZE = args.batch_size
    conf.TRAIN_FILENAME = args.train_path
    conf.TEST_FILENAME = args.test_path
    conf.IS_PLOT = args.plot
    conf.LEARNING_RATE = args.learning_rate

    assert 0 < conf.VALID_SIZE < 1
    assert conf.EPOCHS > 0
    assert conf.BATCH_SIZE > 0
    assert conf.LEARNING_RATE > 0

    if args.train:
        train()
    elif args.predict:
        predict()
    else:
        parser.print_help()
        sys.exit()


if __name__ == "__main__":
    main()