from .distribution import Distribution


class ExponentialFamily(Distribution):
    r"""
    ExponentialFamily inherits from Distribution. ExponentialFamily is a base
    class for distributions whose density function has the form:
    p_F(x;\theta) = exp(
        <t(x), \theta> - 
        F(\theta) + 
        k(x)
    ) where
    t(x): sufficient statistics
    \theta: natural parameters
    F(\theta): log_normalizer
    k(x): carrier measure
    """

    @property
    def _natural_params(self):
        r"""
        Return a tuple that stores natural parameters of the distribution.
        """
        raise NotImplementedError

    @property
    def _log_normalizer(self):
        r"""
        Return the log_normalizer F(\theta) based the natural parameters.
        """
        raise NotImplementedError

    @property
    def _mean_carrier_measure(self, x):
        r"""
        Return the mean of carrier measure k(x) based on input x,
        this method is required for calculating the entropy.
        """
        raise NotImplementedError

    @property
    def entropy(self):
        r"""
        Return the entropy of a distribution.
        The entropy of distributions in exponential families
        could be computed by:
        H(P) = F(\theta) - <\theta, F(\theta)'> - E_p[k(x)]
        """
        # TODO
        raise NotImplementedError

