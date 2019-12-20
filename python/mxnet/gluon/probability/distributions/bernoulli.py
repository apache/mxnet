from mxnet import np, npx
from .exp_family import ExponentialFamily
from .distribution import getF
from .util import prob2logit, logit2prob
import math


class Bernoulli(ExponentialFamily):
    r"""Create a bernoulli

    Parameters
    ----------
    ExponentialFamily : [type]
        [description]
    """

    def __init__(self, prob=None, logit=None, F=None):
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
        return self._prob if self._prob is not None else logit2prob(self._logit)

    def logit(self):
        return self._logit if self._logit is not None else prob2logit(self._prob)

    @property
    def mean(self):
        return self.prob

    @property
    def variance(self):
        return self.prob * (1 - self.prob)

    def log_prob(self, value):
        if (self._prob is None):
            # Parameterized by logit
            relu = 
            return 
        else:
            # Parameterized by probability
            eps = 1e-12
            return (np.log(self._prob + eps) * value
                    + np.log1p(-self._prob + eps) * (1 - value))

    def sample(self):
        pass

    @property
    def _natural_params(self):
        pass

    @property
    def _log_normalizer(self, x):
        pass
