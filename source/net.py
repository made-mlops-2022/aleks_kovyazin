import numpy as np
import json, pickle
import warnings
from sklearn.preprocessing import MinMaxScaler
from typing import List, Dict
from abc import abstractmethod, ABC
from loguru import logger

LEARNING_RATE = 0.001

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)

def cross_entropy(probability, y_true):
    y_pred = probability[np.arange(len(y_true)), y_true]
    return -np.mean(np.log(y_pred))

class LayerABC(ABC):
    def __init__(self):
        self.params = {}

    @classmethod
    @abstractmethod
    def import_layer(cls, dct: Dict):
        logger.info(f'Import layer {cls.__name__} with params: {dct["params"]}')
        return cls(**dct['params'])

    @abstractmethod
    def export_layer(self) -> Dict:
        logger.info(f'Export layer {self.__class__.__name__}')
        return {'type': self.__class__.__name__, 'params': self.params}

class Relu(LayerABC):
    def forward(self, input):
        relu_forward = np.maximum(0, input)
        return relu_forward

    def backward(self, input, grad_output):
        relu_grad = input > 0
        return grad_output * relu_grad

    @classmethod
    def import_layer(cls, dct: Dict):
        return super().import_layer(dct)

    def export_layer(self) -> Dict:
        return super().export_layer()

class Linear(LayerABC):
    def __init__(self, in_feat, out_feat, lr=LEARNING_RATE):
        super().__init__()
        self.params.update({'in_feat': in_feat, 'out_feat': out_feat,
                            'lr': lr})
        self.learning_rate = lr
        self.weights = np.random.normal(loc=0.0,
                        scale=np.sqrt(1 / (in_feat + out_feat)),
                        size=(in_feat, out_feat))
        self.biases = np.zeros(out_feat)

    @classmethod
    def import_layer(cls, dct: Dict):
        layer = super().import_layer(dct)
        if 'weight' in dct and 'biases' in dct:
            layer.weights = np.array(dct['weights'])
            layer.biases = np.array(dct['biases'])
        return layer

    def export_layer(self) -> Dict:
        config_layer = super().export_layer()
        config_layer.update({'weights': self.weights.tolist(), 'biases': self.biases.tolist()})
        return config_layer

    def forward(self, input):
        return np.dot(input, self.weights) + self.biases

    def backward(self, input, grad_output):
        grad_input = np.dot(grad_output, self.weights.T)
        grad_weights = np.dot(input.T, grad_output)
        grad_biases = grad_output.mean(axis=0) * input.shape[0]

        self.weights = self.weights - self.learning_rate * grad_weights
        self.biases = self.biases - self.learning_rate * grad_biases

        return grad_input

class NN:
    layers = {Relu.__name__: Relu,
              Linear.__name__: Linear}
    def __init__(self):
        self.best_weights = []
        self.best_biases = []
        self.best_loss = np.inf
        self.best_epoch = 0
        self.layers = []
        self.scaler = MinMaxScaler()

    def __len__(self):
        return len(self.layers)

    def __iter__(self):
        return iter(self.layers)

    def __getitem__(self, item):
        return self.layers.__getitem__(item)

    def scale_fit_transform(self, X):
        return self.scaler.fit_transform(X)

    def scale_transform(self, X):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            try:
                return self.scaler.transform(X)
            except Exception as e:
                logger.error(f'Scaler transform error {repr(e)}')
                return  self.scaler.fit_transform(X)

    @classmethod
    def import_nn(cls, filename):
        nn = cls()
        try:
            with open(filename) as f:
                conf = json.load(f)
            for layer_conf in conf['layers']:
                layer = cls.layers[layer_conf['type']].import_layer(layer_conf)
                nn.append(layer)
            if 'scaler' in conf['model']:
                nn.scaler = pickle.loads(bytes(conf['model']['scaler'], encoding='latin1'))
            else:
                nn.scaler = MinMaxScaler()
            return nn

        except Exception as e:
            logger.error(f'Scaler import error {repr(e)}')
            raise e

    def export_nn(self, filename):
        try:
            conf = {'model': {'scaler': str(pickle.dumps(self.scaler),
                              encoding='latin1')}, 'layers': []}
            for layer in self.layers:
                conf['layers'].append(layer.export_layer())
            with open(filename, 'w') as f:
                json.dump(conf, f, indent=4, ensure_ascii=False)
        except Exception as e:
            logger.error(f'Scaler export error {repr(e)}')
            raise e

    def train(self, input, targets):
        layer_activations = self.forward(input)
        layer_inputs = [input] + layer_activations
        logits = layer_activations[-1]

        probability = self.predict_proba(logits)
        loss = cross_entropy(probability, targets)

        answers_matrix = np.zeros_like(logits)
        answers_matrix[np.arange(len(logits)), targets] = 1
        loss_grad = (softmax(logits) - answers_matrix) / logits.shape[0]
        for layer_index in range(len(self) - 1, -1, -1):
            layer = self[layer_index]
            loss_grad = layer.backward(layer_inputs[layer_index], loss_grad)
        return loss

    def append(self, layer: LayerABC):
        self.layers.append(layer)

    def predict_proba(self, logits):
        return softmax(logits)

    def loss(self, X, y):
        return cross_entropy(self.predict_proba(self.forward(X)[-1]), y)

    def forward(self, input):
        activations = []
        for el in self:
            activations.append(el.forward(input))
            input = activations[-1]

        return activations

    def memorize_best_weights(self, loss, epoch):
        if loss < self.best_loss:
            self.best_epoch = epoch
            self.best_loss = loss
            self.best_weights = [layer.weights for layer in self if isinstance(layer, Linear)]
            self.best_biases = [layer.biases for layer in self if isinstance(layer, Linear)]

    def use_best_weights(self):
        logger.info('Best weight loaded')
        best_weights = iter(self.best_weights)
        best_bias = iter(self.best_biases)
        for layer in self:
            if isinstance(layer, Linear):
                layer.weights = next(best_weights)
                layer.biases = next(best_bias)