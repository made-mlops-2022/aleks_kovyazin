from pathlib import Path
LEARNING_RATE = 0.001
MODEL_DATA_PATH = f'{Path(__file__).resolve(strict=True).parent.parent}' + '/data/model.json'
TRAIN_FILENAME =f'{Path(__file__).resolve(strict=True).parent.parent}' + '/data/data.csv'
TEST_FILENAME = f'{Path(__file__).resolve(strict=True).parent.parent}' + '/data/data_test.csv'
FILE_LOG_PATH = f'{Path(__file__).resolve(strict=True).parent.parent}' + '/file.log'
IS_PLOT = False
BATCH_SIZE = 2
VALID_SIZE = 0.1
EPOCHS = 10
EPS = 1e-12