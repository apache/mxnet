from .distribution import Distribution

class MultivariateNormal(Distribution):
    def __init__(self, loc, cov=None, precision=None,)