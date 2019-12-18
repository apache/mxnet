from .distribution import Distribution


class ExponentialFamily(Distribution):
    r"""
    p_F(x;\theta) = exp(
        <t(x), \theta> - 
        F(\theta) + 
        k(x)
    )
    t(x): sufficient statistics
    \theta: natural parameters
    F(\theta): log_normalizer
    k(x): carrier measure
    """

    @property
    def _natural_params(self):
        raise NotImplementedError

    @property
    def _log_normalizer(self, *nparams):
        raise NotImplementedError

    @property
    def _mean_carrier_measure(self, x):
        raise NotImplementedError

    @property
    def entropy(self):
        r"""
        H(P) = F(\theta) - <\theta, F(\theta)'> - E_p[k(x)]
        """
        raise NotImplementedError
