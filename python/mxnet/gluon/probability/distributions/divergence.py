# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

# coding: utf-8
# pylint: disable=wildcard-import
"""KL divergence functions."""
__all__ = ['register_kl', 'kl_divergence', 'empirical_kl']

import math
import numpy as _np

from .utils import gammaln, digamma
from .exponential import Exponential
from .pareto import Pareto
from .uniform import Uniform
from .normal import Normal
from .laplace import Laplace
from .cauchy import Cauchy
from .poisson import Poisson
from .geometric import Geometric
from .gamma import Gamma
from .dirichlet import Dirichlet
from .beta import Beta
from .half_normal import HalfNormal
from .bernoulli import Bernoulli
from .binomial import Binomial
from .gumbel import Gumbel
from .categorical import Categorical
from .one_hot_categorical import OneHotCategorical
from .multivariate_normal import MultivariateNormal


def empirical_kl(p, q, n_samples=1):
    r"""Estimate KL(p||q) through monte-carlo estimation, i.e. approximate
    KL(p||q) with:

        1/M * \Sum_{i=1}^{M} log(p(x_i) / q(x_i)), x_i ~ p(x)

    Parameters
    ----------
    p : Distribution
    q : Distribution
    n_samples : int, optional
        Number of monte-carlo samples, by default 1
    """
    samples = p.sample_n(n_samples)
    return (p.log_prob(samples) - q.log_prob(samples)).mean(0)


def register_kl(typeP, typeQ):
    """Decorator for registering custom implementation of kl divergence between
    distribution `typeP` and `typeQ`

    Returns
    ------- function
    """
    func_name = "_kl_" + str(typeP.__name__) \
                + "_" + str(typeQ.__name__)

    def decorator(func):
        func_arg_num = func.__code__.co_argcount
        if (func_arg_num != 2):
            raise TypeError('Expect kl_divergence implementation '
                            + 'to have exactly two arguments, but got {}'.format(func_arg_num))
        if not hasattr(_KL_storage, func_name):
            setattr(_KL_storage, func_name, func)
        else:
            # Behavior TBD.
            print("Error: Duplicate definition")
        return func
    return decorator


def kl_divergence(p, q):
    r"""
    Return the kl divergence between p and q,
    this method will automatically dispatch
    to the corresponding function based on q's type.

    Parameters
    ----------
    p : Distribution
        lhs distribution.
    q : Distribution
        rhs distribution.

    Returns
    -------
    Tensor
        KL(p||q)
    """
    func = _dispatch_kl(p.__class__.__name__, q.__class__.__name__)
    return func(p, q) # pylint: disable=not-callable


def _dispatch_kl(type_p, type_q):
    r"""KL divergence methods should be registered
    with distribution name,
    i.e. the implementation of KL(P(\theta)||Q(\theta))
    should be named after _kl_{P}_{Q}

    Parameters
    ----------
    type_q : Typename of a distribution
    type_q : Typename of a distribution


    Returns
    -------
    Get a class method with function name.
    """
    func_name = "_kl_" + str(type_p) + "_" + str(type_q)
    func_impl = getattr(_KL_storage, func_name, None)
    if (not callable(func_impl)):
        raise NotImplementedError(
            "KL divergence between {} and {} is not implemented.".format(type_p, type_q))
    return func_impl


class _KL_storage():
    r"""Class for storing the definition of kl divergence
    between distributions.
    All the class methods should be static
    """

    @staticmethod
    def _kl_Normal_Normal(p, q):
        F = p.F
        var_ratio = (p.scale / q.scale) ** 2
        t1 = ((p.loc - q.loc) / q.scale) ** 2
        return 0.5 * (var_ratio + t1 - 1 - F.np.log(var_ratio))


@register_kl(Bernoulli, Bernoulli)
def _kl_bernoulli_bernoulli(p, q):
    F = p.F
    log_fn = F.np.log
    prob_p = p.prob
    prob_q = q.prob
    t1 = prob_p * log_fn(prob_p / prob_q)
    t2 = (1 - prob_p) * log_fn((1 - prob_p) / (1 - prob_q))
    return t1 + t2


@register_kl(Categorical, Categorical)
def _kl_categorical_categorical(p, q):
    return (p.prob * (p.logit - q.logit)).sum(-1)


@register_kl(OneHotCategorical, OneHotCategorical)
def _kl_onehotcategorical_onehotcategorical(p, q):
    return _kl_categorical_categorical(p._categorical, q._categorical)


@register_kl(Uniform, Uniform)
def _kl_uniform_uniform(p, q):
    F = p.F
    result = F.np.log((q.high - q.low) / (p.high - p.low))
    result = F.np.where((q.low > p.low) | (q.high < p.high), _np.inf, result)
    return result


@register_kl(Cauchy, Cauchy)
def _kl_cauchy_cauchy(p, q):
    F = p.F
    t1 = F.np.log((p.scale + q.scale) ** 2 + (p.loc - q.loc) ** 2)
    t2 = F.np.log(4 * p.scale * q.scale)
    return t1 - t2


@register_kl(Laplace, Laplace)
def _kl_laplace_laplace(p, q):
    F = p.F
    scale_ratio = p.scale / q.scale
    loc_abs_diff = F.np.abs(p.loc - q.loc)
    t1 = -F.np.log(scale_ratio)
    t2 = loc_abs_diff / q.scale
    t3 = scale_ratio * F.np.exp(-loc_abs_diff / p.scale)
    return t1 + t2 + t3 - 1


@register_kl(Poisson, Poisson)
def _kl_poisson_poisson(p, q):
    F = p.F
    t1 = p.rate * (F.np.log(p.rate) - F.np.log(q.rate))
    t2 = (p.rate - q.rate)
    return t1 - t2


@register_kl(Geometric, Geometric)
def _kl_geometric_geometric(p, q):
    F = p.F
    return (-p.entropy() - F.np.log1p(-q.prob) / p.prob - q.logit)


@register_kl(Exponential, Exponential)
def _kl_exponential_exponential(p, q):
    F = p.F
    scale_ratio = p.scale / q.scale
    t1 = -F.np.log(scale_ratio)
    return t1 + scale_ratio - 1


@register_kl(Pareto, Pareto)
def _kl_pareto_pareto(p, q):
    F = p.F
    scale_ratio = p.scale / q.scale
    alpha_ratio = q.alpha / p.alpha
    t1 = q.alpha * F.np.log(scale_ratio)
    t2 = -F.np.log(alpha_ratio)
    result = t1 + t2 + alpha_ratio - 1
    result = F.np.where(p.support._lower_bound <
                        q.support._lower_bound, _np.nan, result)
    return result


@register_kl(Gumbel, Gumbel)
def _kl_gumbel_gumbel(p, q):
    F = p.F
    lgamma = gammaln(F)
    _euler_gamma = _np.euler_gamma
    ct1 = p.scale / q.scale
    ct2 = q.loc / q.scale
    ct3 = p.loc / q.scale
    t1 = -F.np.log(ct1) - ct2 + ct3
    t2 = ct1 * _euler_gamma
    t3 = F.np.exp(ct2 + lgamma(1 + ct1) - ct3)
    return t1 + t2 + t3 - (1 + _euler_gamma)


@register_kl(Gamma, Gamma)
def _kl_gamma_gamma(p, q):
    F = p.F
    lgamma = gammaln(F)
    dgamma = digamma(F)
    return (
        q.shape * F.np.log(q.scale / p.scale) +
        lgamma(q.shape) - lgamma(p.shape) +
        (p.shape - q.shape) * dgamma(p.shape) +
        (p.shape * p.scale) * (1 / q.scale - 1 / p.scale)
    )


@register_kl(Beta, Beta)
def _kl_beta_beta(p, q):
    F = p.F
    lgamma = gammaln(F)
    dgamma = digamma(F)
    sum_params_p = p.beta + p.alpha
    sum_params_q = q.beta + q.alpha
    t1 = lgamma(q.alpha) + lgamma(q.beta) + lgamma(sum_params_p)
    t2 = lgamma(p.alpha) + lgamma(p.beta) + lgamma(sum_params_q)
    t3 = (p.beta - q.beta) * dgamma(p.beta)
    t4 = (p.alpha - q.alpha) * dgamma(p.alpha)
    t5 = (sum_params_q - sum_params_p) * dgamma(sum_params_p)
    return t1 - t2 + t3 + t4 + t5

# http://bariskurt.com/kullback-leibler-divergence-between-two-dirichlet-and-beta-distributions/


@register_kl(Dirichlet, Dirichlet)
def _kl_dirichlet_dirichlet(p, q):
    F = p.F
    lgamma = gammaln(F)
    dgamma = digamma(F)
    sum_p_concentration = p.alpha.sum(-1)
    sum_q_concentration = q.alpha.sum(-1)
    t1 = lgamma(sum_p_concentration) - lgamma(sum_q_concentration)
    t2 = (lgamma(p.alpha) - lgamma(q.alpha)).sum(-1)
    t3 = p.alpha - q.alpha
    t4 = dgamma(p.alpha) - F.np.expand_dims(dgamma(sum_p_concentration), -1)
    return t1 - t2 + (t3 * t4).sum(-1)


@register_kl(HalfNormal, HalfNormal)
def _kl_halfNormal_halfNormal(p, q):
    F = p.F
    var_ratio = (p.scale / q.scale) ** 2
    t1 = ((p.loc - q.loc) / q.scale) ** 2
    return 0.5 * (var_ratio + t1 - 1 - F.np.log(var_ratio))


@register_kl(Binomial, Binomial)
def _kl_binomial_binomial(p, q):
    F = p.F
    kl = p.n * (p.prob * (p.logit - q.logit) +
                F.np.log1p(-p.prob) - F.np.log1p(-q.prob))
    kl = F.np.where(p.n > q.n, _np.inf, kl)
    return kl


@register_kl(MultivariateNormal, MultivariateNormal)
def _kl_mvn_mvn(p, q):
    F = p.F
    log_det = (lambda mvn:
               F.np.log(
                   F.np.diagonal(mvn.scale_tril, axis1=-2, axis2=-1)
               ).sum(-1)
               )
    # log(det(\Sigma_1) / det(\Sigma_2))
    term1 = log_det(q) - log_det(p)

    # tr(inv(\Sigma_2) * \Sigma_1)
    term2 = F.np.trace(F.np.matmul(q.precision, p.cov), axis1=-2, axis2=-1)

    # (\mu_2 - \mu_1).T * inv(\Sigma_2) * (\mu_2 - \mu_1)
    diff = q.loc - p.loc
    term3 = F.np.einsum(
        '...i,...i->...',
        diff,
        # Batch matrix vector multiply
        F.np.einsum('...jk,...j->...k', q.precision, diff)
    ) * -0.5
    n = F.np.ones_like(diff).sum(-1)
    return 0.5 * (term1 + term2 + term3 - n)


@register_kl(Uniform, Normal)
def _kl_uniform_normal(p, q):
    F = p.F
    common_term = p.high - p.low
    t1 = F.np.log(math.sqrt(math.pi * 2) * q.scale / common_term)
    t2 = (common_term) ** 2 / 12
    t3 = ((p.high + p.low - 2 * q.loc) / 2) ** 2
    return t1 + 0.5 * (t2 + t3) / (q.scale ** 2)


@register_kl(Uniform, Gumbel)
def _kl_uniform_gumbel(p, q):
    F = p.F
    common_term = q.scale / (p.high - p.low)
    high_loc_diff = (p.high - q.loc) / q.scale
    low_loc_diff = (p.low - q.loc) / q.scale
    t1 = F.np.log(common_term) + 0.5 * (high_loc_diff + low_loc_diff)
    t2 = common_term * (F.np.exp(-high_loc_diff) - F.np.exp(-low_loc_diff))
    return t1 - t2


@register_kl(Exponential, Gumbel)
def _kl_exponential_gumbel(p, q):
    F = p.F
    scale_rate_prod = q.scale / p.scale
    loc_scale_ratio = q.loc / q.scale
    t1 = F.np.log(scale_rate_prod) - 1
    t2 = F.np.exp(loc_scale_ratio) * scale_rate_prod / (scale_rate_prod + 1)
    t3 = scale_rate_prod ** -1
    return t1 - loc_scale_ratio + t2 + t3


@register_kl(Exponential, Normal)
def _kl_exponential_normal(p, q):
    F = p.F
    var_normal = q.variance
    rate_sqr = p.scale ** (-2)
    t1 = 0.5 * F.np.log(rate_sqr * var_normal * 2 * _np.pi)
    t2 = rate_sqr ** -1
    t3 = q.loc * p.scale
    t4 = (q.loc ** 2) * 0.5
    return t1 - 1 + (t2 - t3 + t4) / var_normal


@register_kl(Exponential, Gamma)
def _kl_exponential_gamma(p, q):
    F = p.F
    lgamma = gammaln(F)
    ratio = p.scale / q.scale
    t1 = -q.shape * F.np.log(ratio)
    return t1 + ratio + lgamma(q.shape) + q.shape * _np.euler_gamma - (1 + _np.euler_gamma)
