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

"""Random distribution generator Symbol API of MXNet."""

from ..base import numeric_types, _Null
from . import _internal
from .symbol import Symbol


__all__ = ['uniform', 'normal', 'poisson', 'exponential', 'gamma', 'multinomial',
           'negative_binomial', 'generalized_negative_binomial']


def _random_helper(random, sampler, params, shape, dtype, kwargs):
    """Helper function for random generators."""
    if isinstance(params[0], Symbol):
        for i in params[1:]:
            assert isinstance(i, Symbol), \
                "Distribution parameters must all have the same type, but got " \
                "both %s and %s."%(type(params[0]), type(i))
        return sampler(*params, shape=shape, dtype=dtype, **kwargs)
    elif isinstance(params[0], numeric_types):
        for i in params[1:]:
            assert isinstance(i, numeric_types), \
                "Distribution parameters must all have the same type, but got " \
                "both %s and %s."%(type(params[0]), type(i))
        return random(*params, shape=shape, dtype=dtype, **kwargs)

    raise ValueError("Distribution parameters must be either Symbol or numbers, "
                     "but got %s."%type(params[0]))


def uniform(low=0, high=1, shape=_Null, dtype=_Null, **kwargs):
    """Draw random samples from a uniform distribution.

    Samples are uniformly distributed over the half-open interval *[low, high)*
    (includes *low*, but excludes *high*).

    Parameters
    ----------
    low : float or Symbol
        Lower boundary of the output interval. All values generated will be
        greater than or equal to low. The default value is 0.
    high : float or Symbol
        Upper boundary of the output interval. All values generated will be
        less than high. The default value is 1.0.
    shape : int or tuple of ints
        The number of samples to draw. If shape is, e.g., `(m, n)` and `low` and
        `high` are scalars, output shape will be `(m, n)`. If `low` and `high`
        are Symbols with shape, e.g., `(x, y)`, then output will have shape
        `(x, y, m, n)`, where `m*n` samples are drawn for each `[low, high)` pair.
    dtype : {'float16','float32', 'float64'}
        Data type of output samples. Default is 'float32'
    """
    return _random_helper(_internal._random_uniform, _internal._sample_uniform,
                          [low, high], shape, dtype, kwargs)


def normal(loc=0, scale=1, shape=_Null, dtype=_Null, **kwargs):
    """Draw random samples from a normal (Gaussian) distribution.

    Samples are distributed according to a normal distribution parametrized
    by *loc* (mean) and *scale* (standard deviation).


    Parameters
    ----------
    loc : float or Symbol
        Mean (centre) of the distribution.
    scale : float or Symbol
        Standard deviation (spread or width) of the distribution.
    shape : int or tuple of ints
        The number of samples to draw. If shape is, e.g., `(m, n)` and `loc` and
        `scale` are scalars, output shape will be `(m, n)`. If `loc` and `scale`
        are Symbols with shape, e.g., `(x, y)`, then output will have shape
        `(x, y, m, n)`, where `m*n` samples are drawn for each `[loc, scale)` pair.
    dtype : {'float16','float32', 'float64'}
        Data type of output samples. Default is 'float32'
    """
    return _random_helper(_internal._random_normal, _internal._sample_normal,
                          [loc, scale], shape, dtype, kwargs)


def poisson(lam=1, shape=_Null, dtype=_Null, **kwargs):
    """Draw random samples from a Poisson distribution.

    Samples are distributed according to a Poisson distribution parametrized
    by *lambda* (rate). Samples will always be returned as a floating point data type.

    Parameters
    ----------
    lam : float or Symbol
        Expectation of interval, should be >= 0.
    shape : int or tuple of ints
        The number of samples to draw. If shape is, e.g., `(m, n)` and `lam` is
        a scalar, output shape will be `(m, n)`. If `lam`
        is an Symbol with shape, e.g., `(x, y)`, then output will have shape
        `(x, y, m, n)`, where `m*n` samples are drawn for each entry in `lam`.
    dtype : {'float16','float32', 'float64'}
        Data type of output samples. Default is 'float32'
    """
    return _random_helper(_internal._random_poisson, _internal._sample_poisson,
                          [lam], shape, dtype, kwargs)


def exponential(scale=1, shape=_Null, dtype=_Null, **kwargs):
    r"""Draw samples from an exponential distribution.

    Its probability density function is

        f(x; \frac{1}{\beta}) = \frac{1}{\beta} \exp(-\frac{x}{\beta}),

    for x > 0 and 0 elsewhere. \beta is the scale parameter, which is the
    inverse of the rate parameter \lambda = 1/\beta.

    Parameters
    ----------
    scale : float or Symbol
        The scale parameter, \beta = 1/\lambda.
    shape : int or tuple of ints
        The number of samples to draw. If shape is, e.g., `(m, n)` and `scale` is
        a scalar, output shape will be `(m, n)`. If `scale`
        is an Symbol with shape, e.g., `(x, y)`, then output will have shape
        `(x, y, m, n)`, where `m*n` samples are drawn for each entry in `scale`.
    dtype : {'float16','float32', 'float64'}
        Data type of output samples. Default is 'float32'
    """
    return _random_helper(_internal._random_exponential, _internal._sample_exponential,
                          [1.0/scale], shape, dtype, kwargs)


def gamma(alpha=1, beta=1, shape=_Null, dtype=_Null, **kwargs):
    """Draw random samples from a gamma distribution.

    Samples are distributed according to a gamma distribution parametrized
    by *alpha* (shape) and *beta* (scale).

    Parameters
    ----------
    alpha : float or Symbol
        The shape of the gamma distribution. Should be greater than zero.
    beta : float or Symbol
        The scale of the gamma distribution. Should be greater than zero.
        Default is equal to 1.
    shape : int or tuple of ints
        The number of samples to draw. If shape is, e.g., `(m, n)` and `alpha` and
        `beta` are scalars, output shape will be `(m, n)`. If `alpha` and `beta`
        are Symbols with shape, e.g., `(x, y)`, then output will have shape
        `(x, y, m, n)`, where `m*n` samples are drawn for each `[alpha, beta)` pair.
    dtype : {'float16','float32', 'float64'}
        Data type of output samples. Default is 'float32'
    """
    return _random_helper(_internal._random_gamma, _internal._sample_gamma,
                          [alpha, beta], shape, dtype, kwargs)


def negative_binomial(k=1, p=1, shape=_Null, dtype=_Null, **kwargs):
    """Draw random samples from a negative binomial distribution.

    Samples are distributed according to a negative binomial distribution
    parametrized by *k* (limit of unsuccessful experiments) and *p* (failure
    probability in each experiment). Samples will always be returned as a
    floating point data type.

    Parameters
    ----------
    k : float or Symbol
        Limit of unsuccessful experiments, > 0.
    p : float or Symbol
        Failure probability in each experiment, >= 0 and <=1.
    shape : int or tuple of ints
        The number of samples to draw. If shape is, e.g., `(m, n)` and `k` and
        `p` are scalars, output shape will be `(m, n)`. If `k` and `p`
        are Symbols with shape, e.g., `(x, y)`, then output will have shape
        `(x, y, m, n)`, where `m*n` samples are drawn for each `[k, p)` pair.
    dtype : {'float16','float32', 'float64'}
        Data type of output samples. Default is 'float32'
    """
    return _random_helper(_internal._random_negative_binomial,
                          _internal._sample_negative_binomial,
                          [k, p], shape, dtype, kwargs)


def generalized_negative_binomial(mu=1, alpha=1, shape=_Null, dtype=_Null, **kwargs):
    """Draw random samples from a generalized negative binomial distribution.

    Samples are distributed according to a generalized negative binomial
    distribution parametrized by *mu* (mean) and *alpha* (dispersion).
    *alpha* is defined as *1/k* where *k* is the failure limit of the
    number of unsuccessful experiments (generalized to real numbers).
    Samples will always be returned as a floating point data type.

    Parameters
    ----------
    mu : float or Symbol
        Mean of the negative binomial distribution.
    alpha : float or Symbol
        Alpha (dispersion) parameter of the negative binomial distribution.
    shape : int or tuple of ints
        The number of samples to draw. If shape is, e.g., `(m, n)` and `mu` and
        `alpha` are scalars, output shape will be `(m, n)`. If `mu` and `alpha`
        are Symbols with shape, e.g., `(x, y)`, then output will have shape
        `(x, y, m, n)`, where `m*n` samples are drawn for each `[mu, alpha)` pair.
    dtype : {'float16','float32', 'float64'}
        Data type of output samples. Default is 'float32'
    """
    return _random_helper(_internal._random_generalized_negative_binomial,
                          _internal._sample_generalized_negative_binomial,
                          [mu, alpha], shape, dtype, kwargs)


def multinomial(data, shape=_Null, get_prob=True, **kwargs):
    """Concurrent sampling from multiple multinomial distributions.

    .. note:: The input distribution must be normalized, i.e. `data` must sum to
              1 along its last dimension.

    Parameters
    ----------
    data : Symbol
        An *n* dimensional array whose last dimension has length `k`, where
        `k` is the number of possible outcomes of each multinomial distribution.
        For example, data with shape `(m, n, k)` specifies `m*n` multinomial
        distributions each with `k` possible outcomes.
    shape : int or tuple of ints
        The number of samples to draw from each distribution. If shape is empty
        one sample will be drawn from each distribution.
    get_prob : bool
        If true, a second array containing log likelihood of the drawn
        samples will also be returned.
        This is usually used for reinforcement learning, where you can provide
        reward as head gradient w.r.t. this array to estimate gradient.
    """
    return _internal._sample_multinomial(data, shape, get_prob, **kwargs)
