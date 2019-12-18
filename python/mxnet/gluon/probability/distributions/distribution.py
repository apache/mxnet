import mxnet as mx
from mxnet import np, npx


def getF(*params):
    for param in params:
        if isinstance(param, np.ndarray):
            return mx.ndarray
        elif isinstance(param, mx.symbol.numpy._Symbol):
            return mx.symbol.numpy._Symbol
    return mx.ndarray

class Distribution:
    has_grad = False

    def __init__(self, F=None):
        self.F = F

    def log_prob(self, x):
        raise NotImplementedError()

    def prob(self, x):
        raise NotImplementedError

    def sample(self, size):
        raise NotImplementedError

    def sample_n(self, batch_size):
        raise NotImplementedError