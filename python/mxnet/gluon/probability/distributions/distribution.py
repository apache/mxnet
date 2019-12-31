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

    def sample(self, size):
        r"""
        Generate samples of `size` from the distribution.
        """
        raise NotImplementedError

    def sample_n(self, batch_size):
        r"""
        Generate samples of (batch_size + parameter_size) from the distribution.
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

    @classmethod
    def _dispatch_kl(cls, type_q):
        r"""KL divergence methods should be registered
        with distribution name,
        i.e. the implementation of KL(P(\theta)||Q(\theta))
        should be named after _kl_{P}_{Q}

        Parameters
        ----------
        type_q : Typename of a distribution
            
        
        Returns
        -------
        Get a class method with function name.
        """
        func_name = "_kl_" + cls.__name__ + "_" + str(type_q)
        return getattr(cls, func_name)

    def kl_divergence(self, q):
        r"""Return the kl divergence with q,
        this method will automatically dispatch
        to the corresponding function based on q's type.
        
        Parameters
        ----------
        q : Distribution
            Target distribution.
        
        Returns
        -------
        Tensor
            KL(self||q)
        """
        kl_func = self._dispatch_kl(q.__class__.__name__)
        return kl_func(self, q)