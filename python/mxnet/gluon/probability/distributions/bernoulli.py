from mxnet import np, npx
from .exp_family import ExponentialFamily
from .util import prob2logit, logit2prob, getF
import math


class Bernoulli(ExponentialFamily):
    r"""Create a bernoulli distribution object.

    Parameters
    ----------
    prob : Tensor or scalar, default 0.5
        Probability of sampling `1`.
    logit : Tensor or scalar, default None
        The log-odds of sampling `1`.
    F : mx.ndarray or mx.symbol.numpy._Symbol or None
        Variable recording running mode, will be automatically
        inferred from parameters if declared None.
    """

    def __init__(self, prob=0.5, logit=None, F=None):
        _F = F if F is not None else getF([prob, logit])
        super(Bernoulli, self).__init__(F=_F)

        if (prob == None) == (logit == None):
            raise ValueError(
                "Either `prob` or `logit` must be specified, but not both. " +
                "Received prob={}, logit={}".format(prob, logit))

        self._prob = prob
        self._logit = logit

    @property
    def prob(self):
        """Get the probability of sampling `1`.
        
        Returns
        -------
        Tensor
            Parameter tensor.
        """        
        return self._prob if self._prob is not None else logit2prob(self._logit, self.F)

    def logit(self):
        """Get the log-odds of sampling `1`.
        
        Returns
        -------
        Tensor
            Parameter tensor.
        """        
        return self._logit if self._logit is not None else prob2logit(self._prob, self.F)

    @property
    def mean(self):
        return self.prob

    @property
    def variance(self):
        return self.prob * (1 - self.prob)

    def log_prob(self, value):
        F = self.F
        if (self._prob is None):
            # When the distribution is parameterized by logit,
            # we apply the formula:
            # max(logit, 0) - value * logit + log(1 + exp(-abs(logit)))
            relu = F.npx.relu
            abs_fn = F.np.abs
            logit = self.logit
            return (relu(logit) - value * logit +
                    F.npx.activation(-abs_fn(logit), act_type='softrelu'))
        else:
            # Parameterized by probability
            eps = 1e-12
            return (self.F.np.log(self._prob + eps) * value
                    + self.F.np.log1p(-self._prob + eps) * (1 - value))

    def sample(self, size=None):
        return self.F.npx.random.bernoulli(self._prob, self._logit, size)

    @property
    def _natural_params(self):
        return (self.logit,)

    @property
    def _log_normalizer(self, x):
        return self.F.np.log(1 + self.F.np.exp(x))
