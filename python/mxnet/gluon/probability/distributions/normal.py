from mxnet import np, npx
from .exp_family import ExponentialFamily
from .util import getF
import math


class Normal(ExponentialFamily):
    r"""Create a normal distribution object.

    Parameters
    ----------
    loc : Tensor or scalar, default 0
        mean of the distribution.
    scale : Tensor or scalar, default 1
        standard deviation of the distribution
    F : mx.ndarray or mx.symbol.numpy._Symbol or None
        Variable recording running mode, will be automatically
        inferred from parameters if declared None.

    """

    has_grad = True

    def __init__(self, loc=0.0, scale=1.0, F=None):
        _F = F if F is not None else getF([loc, scale])
        super(Normal, self).__init__(F=_F)
        self.loc = loc
        self.scale = scale

    @property
    def mean(self):
        """Return the mean of the normal distribution

        Returns
        -------
        Tensor
            A parameter tensor.
        """
        return self.loc

    @property
    def variance(self):
        """Return the variance of the normal distribution.

        Returns
        -------
        Tensor
            Square of `self._scale`
        """
        return self.scale ** 2

    def log_prob(self, value):
        """Compute the log likehood of `value`.

        Parameters
        ----------
        value : Tensor
            Input data.

        Returns
        -------
        Tensor
            Log likehood of the input.
        """
        F = self.F
        var = (self.scale ** 2)
        log_scale = F.np.log(self.scale)
        return (-((value - self.loc) ** 2) / (2 * var) -
                log_scale - F.np.log(F.np.sqrt(2 * math.pi)))

    def sample(self, size=None):
        r"""Generate samples of `size` from the normal distribution
        parameterized by `self.loc` and `self.scale`

        Parameters
        ----------
        size : Tuple, Scalar, or None
            Size of samples to be generated. If size=None, the output shape
            will be `broadcast(loc, scale).shape`

        Returns
        -------
        Tensor
            Samples from Normal distribution.
        """
        return self.F.np.random.normal(self.loc,
                                       self.scale,
                                       size)

    def sample_n(self, batch_size=None):
        r"""Generate samples of (batch_size + broadcast(loc, scale).shape)
        from the normal distribution parameterized by `self.loc` and `self.scale`

        Parameters
        ----------
        batch_size : Tuple, Scalar, or None
            Size of independent batch to be generated from the distribution.

        Returns
        -------
        Tensor
            Samples from Normal distribution.
        """
        return self.F.npx.random.normal_n(self.loc,
                                          self.scale,
                                          batch_size)

    @property
    def _natural_params(self):
        r"""Return the natural parameters of normal distribution,
        which are (\frac{\mu}{\sigma^2}, -0.5 / (\sigma^2))

        Returns
        -------
        Tuple
            Natural parameters of normal distribution.
        """
        return (self.loc / (self.scale ** 2),
                -0.5 * self.F.np.reciprocal(self.scale ** 2))

    @property
    def _log_normalizer(self, x, y):
        """Return the log_normalizer term of normal distribution in exponential family term.

        Parameters
        ----------
        x : Tensor
            The first natural parameter.
        y : Tensor
            The second natural parameter.

        Returns
        -------
        Tensor
            the log_normalizer term
        """
        F = self.F
        return -0.25 * F.np.pow(2) / y + 0.5 * F.np.log(-F.np.pi / y)
