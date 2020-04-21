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

import numpy as _np

from .utils import getF, gammaln
from .exponential import *
from .weibull import *
from .pareto import *
from .uniform import *
from .normal import *
from .laplace import *
from .cauchy import *
from .half_cauchy import *
from .poisson import *
from .geometric import *
from .negative_binomial import *
from .gamma import *
from .dirichlet import *
from .beta import *
from .chi2 import *
from .fishersnedecor import *
from .studentT import *
from .half_normal import *
from .independent import *
from .bernoulli import *
from .binomial import *
from .relaxed_bernoulli import *
from .gumbel import *
from .categorical import *
from .one_hot_categorical import *
from .relaxed_one_hot_categorical import *
from .multinomial import *
from .multivariate_normal import *


def empirical_kl(p, q, n_samples=1):
    """Estimate KL(p||q) through monte-carlo estimation,
    i.e. approximate KL(p||q) with:
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
    """Decorator for registering custom implementation
    of kl divergence between distribution `typeP` and `typeQ`

    Returns
    -------
    wrapped function
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
    return func(p, q)


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
    result[(q.low > p.low) | (q.high < p.high)] = _np.inf
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
    # TODO: Handle out-of-support value
    # result[p.support.lower_bound < q.support.lower_bound] = inf
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


@register_kl(HalfNormal, HalfNormal)
def _kl_halfNormal_halfNormal(p, q):
    F = p.F
    var_ratio = (p.scale / q.scale) ** 2
    t1 = ((p.loc - q.loc) / q.scale) ** 2
    return 0.5 * (var_ratio + t1 - 1 - F.np.log(var_ratio))


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
            F.np.einsum('...jk,...j->...k', q.precision, diff)  # Batch matrix vector multiply
            ) * -0.5
    n = F.np.ones_like(diff).sum(-1)
    return 0.5 * (term1 + term2 + term3 - n)