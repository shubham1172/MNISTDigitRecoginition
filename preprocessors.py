# preprocessors
import numpy as np


def simple_preprocessor(data):
    return np.reshape(data, (1, 28 * 28))


def simple_cnn_preprocessor(data):
    return np.expand_dims(data, axis=0)
