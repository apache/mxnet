from .distribution import Distribution
from .categorical import Categorical
from .utils import getF

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
        super(OneHotCategorical, self).__init__(_F, event_dim=1, validate_args)

    