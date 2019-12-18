from mxnet import np, npx
from .exp_family import ExponentialFamily
from .distribution import getF


class Normal(ExponentialFamily):

    def __init__(self, loc=0.0, scale=1.0, F=None):
        self.F = F if F is not None else getF([loc, scale])
        super(Normal, self).__init__(F=F)
        self._loc = loc
        self._scale = scale
        self.F = F

    def sample(self, size=None):
        return self.F.np.random.normal(self._loc,
                                    self._scale,
                                    size)

    def sample_n(self, batch_size=None):
        return self.F.npx.random.normal_n(self._loc,
                                        self._scale,
                                        batch_size)

    def log_prob(self, value):
        F = self.F
        var = (self._scale ** 2)
        log_scale = F.np.log(self._scale)
        return -((value - self._loc) ** 2) / (2 * var) - log_scale - F.np.log(F.np.sqrt(2 * F.np.pi))
