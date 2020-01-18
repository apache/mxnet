
from ..transformation import Transformation
from .distribution import Distribution
from .utils import getF

class TransformedDistribution(Distribution):
    def __init__(self, base_dist, transforms):
        self._base_dict = base_dist
        if isinstance(transforms, Transformation):
            transforms = [transforms,]
        self._transforms = transforms

        _F = base_dist.F
        # Overwrite the F in transform
        for t in self._transforms:
            t.F = _F
        super(TransformedDistribution, self).__init__(_F)

    def sample(self, size=None):
        x = self._base_dict.sample(size)
        for t in self._transforms:
            x = t(x)
        return x

    def log_prob(self, value):
        pass