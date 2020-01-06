from mxnet import np, npx
from .distribution import Distribution
from .utils import prob2logit, logit2prob, getF


class Categorical(Distribution):
    """Create a categorical distribution object.

    Parameters
    ----------
    Distribution : [type]
        [description]

    Returns
    -------
    [type]
        [description]

    Raises
    ------
    ValueError
        [description]
    """

    def __init__(self, prob, logit=None, F=None):
        _F = F if F is not None else getF([prob, logit])
        super(Categorical, self).__init__(F=_F)

        if (prob is None) == (logit is None):
            raise ValueError(
                "Either `prob` or `logit` must be specified, but not both. " +
                "Received prob={}, logit={}".format(prob, logit))

        self._prob = prob
        self._logit = logit

    @property
    def prob(self):
        """Get the probability of sampling each class.

        Returns
        -------
        Tensor
            Parameter tensor.
        """
        return self._prob if self._prob is not None else logit2prob(self._logit, False, self.F)

    @property
    def logit(self):
        """Get the log probability of sampling each class.

        Returns
        -------
        Tensor
            Parameter tensor.
        """
        return self._logit if self._logit is not None else prob2logit(self._prob, False, self.F)

    @property
    def mean(self):
        pass

    @property
    def variance(self):
        pass

    def log_prob(self, value):
        pass

    def sample(self, size):
        pass
