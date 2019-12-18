import mxnet as mx
from mxnet import np, npx


def _getF(*params):
    for param in params:
        if isinstance(param, np.ndarray):
            return mx.ndarray
        elif isinstance(param, mx.symbol.numpy._Symbol):
            return mx.symbol.numpy._Symbol
    return mx.ndarray


class Distribution(object):
    has_grad = False

    def __init__(self, F=None):
        self.F = F

    def log_prob(self, x):
        """
        Returns the log of the probability density/mass function evaluated at `x`.
        """
        raise NotImplementedError()

    def prob(self, x):
        """
        Returns the probability density/mass function evaluated at `x`.
        """
        raise NotImplementedError

    def sample(self, shape):
        """
        Generates a `shape` shaped sample.
        """
        raise NotImplementedError

    def sample_n(self, n):
        """
        As MxNet does not support symbolic shape, the following
        code cannot be hybridized.
            normal = Normal(loc=loc, scale=scale)
            samples = normal.sample(size=(n,) + loc.shape)
        Thus, we introduce another function sample_n that infers loc.shape in the backend.
        """
        raise NotImplementedError
