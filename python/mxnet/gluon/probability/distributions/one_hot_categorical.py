__all__ = ['OneHotCategorical']

from .distribution import Distribution
from .categorical import Categorical
from .utils import getF, cached_property

class OneHotCategorical(Distribution):
    def __init__(self, num_events, prob=None, logit=None, F=None, validate_args=None):
        _F = F if F is not None else getF([prob, logit])
        if (num_events > 0):
            num_events = int(num_events)
            self.num_events = num_events
        else:
            raise ValueError("`num_events` should be greater than zero. " +
                             "Received num_events={}".format(num_events))
        self._categorical = Categorical(prob, logit, _F)
        super(OneHotCategorical, self).__init__(_F, event_dim=1, validate_args=validate_args)

    @cached_property
    def prob(self):
        return self._categorical.prob

    @cached_property
    def logit(self):
        return self._categorical.logit

    @property
    def mean(self):
        return self._categorical.prob

    @property
    def variance(self):
        prob = self.prob
        return prob * (1 - prob)

    def sample(self, size=None):
        indices = self._categorical.sample(size)
        return self.F.npx.one_hot(indices, self.num_events)
    
    def log_prob(self, value):
        indices = value.argmax(-1)
        return self._categorical.log_prob(indices)