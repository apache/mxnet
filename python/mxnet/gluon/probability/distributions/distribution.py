import mxnet as mx
from mxnet import np, npx


class Distribution:
    r"""Base class for distribution.
    
    Parameters
    ----------
    F : mx.ndarray or mx.symbol.numpy._Symbol
        Variable that stores the running mode.
    """          

    # Variable indicating whether the sampling method has
    # pathwise gradient.
    has_grad = False

    def __init__(self, F=None):
        self._kl_dict = {}
        self.F = F

    def log_prob(self, x):
        r"""
        Return the log likehood given input x.
        """
        raise NotImplementedError()

    def prob(self, x):
        r"""
        Return the density given input x.
        """
        raise NotImplementedError

    def sample(self, size=None):
        r"""
        Generate samples of `size` from the distribution.
        """
        raise NotImplementedError

    def sample_n(self, batch_size):
        r"""
        Generate samples of (batch_size + parameter_size) from the distribution.
        """
        raise NotImplementedError

    def broadcast_to(self, batch_shape):
        """
        Returns a new distribution instance with parameters expanded
        to `batch_shape`. This method calls `numpy.broadcast_to` on
        the parameters.

        Parameters
        ----------
        new_batch_size : Tuple
            The batch shape of the desired distribution.

        """
        raise NotImplementedError

    def enumerate_support(self):
        r"""
        Returns a tensor that contains all values supported
        by a discrete distribution.
        """
        raise NotImplementedError

    @property
    def mean(self):
        r"""
        Return the mean of the distribution.
        """
        raise NotImplementedError

    @property
    def variance(self):
        r"""
        Return the variance of the distribution.
        """
        return NotImplementedError
