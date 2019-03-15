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
           'negative_binomial', 'generalized_negative_binomial', 'shuffle', 'randint']


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
    low : float or Symbol, optional
        Lower boundary of the output interval. All values generated will be
        greater than or equal to low. The default value is 0.
    high : float or Symbol, optional
        Upper boundary of the output interval. All values generated will be
        less than high. The default value is 1.0.
    shape : int or tuple of ints, optional
        The number of samples to draw. If shape is, e.g., `(m, n)` and `low` and
        `high` are scalars, output shape will be `(m, n)`. If `low` and `high`
        are Symbols with shape, e.g., `(x, y)`, then output will have shape
        `(x, y, m, n)`, where `m*n` samples are drawn for each `[low, high)` pair.
    dtype : {'float16', 'float32', 'float64'}, optional
        Data type of output samples. Default is 'float32'

    Returns
    -------
    Symbol
        If input `shape` has dimensions, e.g., `(m, n)`, and `low` and `high` are
        scalars, returned Symbol will resolve to shape `(m, n)`. If `low` and `high`
        are Symbols with shape, e.g., `(x, y)`, returned Symbol will have shape
        `(x, y, m, n)`, where `m*n` samples are drawn for each `[low, high)` pair.
    """
    return _random_helper(_internal._random_uniform, _internal._sample_uniform,
                          [low, high], shape, dtype, kwargs)


def normal(loc=0, scale=1, shape=_Null, dtype=_Null, **kwargs):
    """Draw random samples from a normal (Gaussian) distribution.

    Samples are distributed according to a normal distribution parametrized
    by *loc* (mean) and *scale* (standard deviation).


    Parameters
    ----------
    loc : float or Symbol, optional
        Mean (centre) of the distribution.
    scale : float or Symbol, optional
        Standard deviation (spread or width) of the distribution.
    shape : int or tuple of ints, optional
        The number of samples to draw. If shape is, e.g., `(m, n)` and `loc` and
        `scale` are scalars, output shape will be `(m, n)`. If `loc` and `scale`
        are Symbols with shape, e.g., `(x, y)`, then output will have shape
        `(x, y, m, n)`, where `m*n` samples are drawn for each `[loc, scale)` pair.
    dtype : {'float16', 'float32', 'float64'}, optional
        Data type of output samples. Default is 'float32'

    Returns
    -------
    Symbol
        If input `shape` has dimensions, e.g., `(m, n)`, and `loc` and
        `scale` are scalars, returned Symbol will resolve to shape `(m, n)`.
        If `loc` and `scale` are Symbols with shape, e.g., `(x, y)`, returned
        Symbol will resolve to shape `(x, y, m, n)`, where `m*n` samples are drawn
        for each `[loc, scale)` pair.
    """
    return _random_helper(_internal._random_normal, _internal._sample_normal,
                          [loc, scale], shape, dtype, kwargs)


def poisson(lam=1, shape=_Null, dtype=_Null, **kwargs):
    """Draw random samples from a Poisson distribution.

    Samples are distributed according to a Poisson distribution parametrized
    by *lambda* (rate). Samples will always be returned as a floating point data type.

    Parameters
    ----------
    lam : float or Symbol, optional
        Expectation of interval, should be >= 0.
    shape : int or tuple of ints, optional
        The number of samples to draw. If shape is, e.g., `(m, n)` and `lam` is
        a scalar, output shape will be `(m, n)`. If `lam`
        is an Symbol with shape, e.g., `(x, y)`, then output will have shape
        `(x, y, m, n)`, where `m*n` samples are drawn for each entry in `lam`.
    dtype : {'float16', 'float32', 'float64'}, optional
        Data type of output samples. Default is 'float32'

    Returns
    -------
    Symbol
        If input `shape` has dimensions, e.g., `(m, n)`, and `lam` is
        a scalar, output shape will be `(m, n)`. If `lam`
        is an Symbol with shape, e.g., `(x, y)`, then output will have shape
        `(x, y, m, n)`, where `m*n` samples are drawn for each entry in `lam`.
    """
    return _random_helper(_internal._random_poisson, _internal._sample_poisson,
                          [lam], shape, dtype, kwargs)


def exponential(scale=1, shape=_Null, dtype=_Null, **kwargs):
    r"""Draw samples from an exponential distribution.

    Its probability density function is

    .. math:: f(x; \frac{1}{\beta}) = \frac{1}{\beta} \exp(-\frac{x}{\beta}),

    for x > 0 and 0 elsewhere. \beta is the scale parameter, which is the
    inverse of the rate parameter \lambda = 1/\beta.

    Parameters
    ----------
    scale : float or Symbol, optional
        The scale parameter, \beta = 1/\lambda.
    shape : int or tuple of ints, optional
        The number of samples to draw. If shape is, e.g., `(m, n)` and `scale` is
        a scalar, output shape will be `(m, n)`. If `scale`
        is an Symbol with shape, e.g., `(x, y)`, then output will have shape
        `(x, y, m, n)`, where `m*n` samples are drawn for each entry in `scale`.
    dtype : {'float16', 'float32', 'float64'}, optional
        Data type of output samples. Default is 'float32'

    Returns
    -------
    Symbol
        If input `shape` has dimensions, e.g., `(m, n)`, and `scale` is
        a scalar, returned Symbol will have shape `(m, n)`. If `scale`
        is a Symbol with shape, e.g., `(x, y)`, returned Symbol will resolve to
        shape `(x, y, m, n)`, where `m*n` samples are drawn for each entry in `scale`.
    """
    return _random_helper(_internal._random_exponential, _internal._sample_exponential,
                          [1.0/scale], shape, dtype, kwargs)


def gamma(alpha=1, beta=1, shape=_Null, dtype=_Null, **kwargs):
    """Draw random samples from a gamma distribution.

    Samples are distributed according to a gamma distribution parametrized
    by *alpha* (shape) and *beta* (scale).

    Parameters
    ----------
    alpha : float or Symbol, optional
        The shape of the gamma distribution. Should be greater than zero.
    beta : float or Symbol, optional
        The scale of the gamma distribution. Should be greater than zero.
        Default is equal to 1.
    shape : int or tuple of ints, optional
        The number of samples to draw. If shape is, e.g., `(m, n)` and `alpha` and
        `beta` are scalars, output shape will be `(m, n)`. If `alpha` and `beta`
        are Symbols with shape, e.g., `(x, y)`, then output will have shape
        `(x, y, m, n)`, where `m*n` samples are drawn for each `[alpha, beta)` pair.
    dtype : {'float16', 'float32', 'float64'}, optional
        Data type of output samples. Default is 'float32'

    Returns
    -------
    Symbol
        If input `shape` has dimensions, e.g., `(m, n)` and `alpha` and
        `beta` are scalars, returned Symbol will resolve to shape `(m, n)`. If `alpha`
        and `beta` are Symbols with shape, e.g., `(x, y)`, returned Symbol will resolve
        to shape `(x, y, m, n)`, where `m*n` samples are drawn for each `[alpha, beta)` pair.
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
    k : float or Symbol, optional
        Limit of unsuccessful experiments, > 0.
    p : float or Symbol, optional
        Failure probability in each experiment, >= 0 and <=1.
    shape : int or tuple of ints
        The number of samples to draw. If shape is, e.g., `(m, n)` and `k` and
        `p` are scalars, output shape will be `(m, n)`. If `k` and `p`
        are Symbols with shape, e.g., `(x, y)`, then output will have shape
        `(x, y, m, n)`, where `m*n` samples are drawn for each `[k, p)` pair.
    dtype : {'float16', 'float32', 'float64'}, optional
        Data type of output samples. Default is 'float32'

    Returns
    -------
    Symbol
        If input `shape` has dimensions, e.g., `(m, n)`, and `k` and
        `p` are scalars, returned Symbol will resolve to shape `(m, n)`. If `k`
        and `p` are Symbols with shape, e.g., `(x, y)`, returned Symbol will resolve
        to shape `(x, y, m, n)`, where `m*n` samples are drawn for each `[k, p)` pair.
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
    mu : float or Symbol, optional
        Mean of the negative binomial distribution.
    alpha : float or Symbol, optional
        Alpha (dispersion) parameter of the negative binomial distribution.
    shape : int or tuple of ints, optional
        The number of samples to draw. If shape is, e.g., `(m, n)` and `mu` and
        `alpha` are scalars, output shape will be `(m, n)`. If `mu` and `alpha`
        are Symbols with shape, e.g., `(x, y)`, then output will have shape
        `(x, y, m, n)`, where `m*n` samples are drawn for each `[mu, alpha)` pair.
    dtype : {'float16', 'float32', 'float64'}, optional
        Data type of output samples. Default is 'float32'

    Returns
    -------
    Symbol
        If input `shape` has dimensions, e.g., `(m, n)`, and `mu` and
        `alpha` are scalars, returned Symbol will resolve to shape `(m, n)`. If `mu`
        and `alpha` are Symbols with shape, e.g., `(x, y)`, returned Symbol will resolve
        to shape `(x, y, m, n)`, where `m*n` samples are drawn for each `[mu, alpha)` pair.
    """
    return _random_helper(_internal._random_generalized_negative_binomial,
                          _internal._sample_generalized_negative_binomial,
                          [mu, alpha], shape, dtype, kwargs)


def multinomial(data, shape=_Null, get_prob=True, dtype='int32', **kwargs):
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
    shape : int or tuple of ints, optional
        The number of samples to draw from each distribution. If shape is empty
        one sample will be drawn from each distribution.
    get_prob : bool, optional
        If true, a second array containing log likelihood of the drawn
        samples will also be returned.
        This is usually used for reinforcement learning, where you can provide
        reward as head gradient w.r.t. this array to estimate gradient.
    dtype : str or numpy.dtype, optional
        Data type of the sample output array. The default is int32.
        Note that the data type of the log likelihood array is the same with that of `data`.

    Returns
    -------
    Symbol
        For input `data` with `n` dimensions and shape `(d1, d2, ..., dn-1, k)`, and input
        `shape` with shape `(s1, s2, ..., sx)`, returns a Symbol that resovles to shape
        `(d1, d2, ... dn-1, s1, s2, ..., sx)`. The `s1, s2, ... sx` dimensions of the
        returned Symbol's resolved value will consist of 0-indexed values sampled from each
        respective multinomial distribution provided in the `k` dimension of `data`.

        For the case `n`=1, and `x`=1 (one shape dimension), returned Symbol will resolve to
        shape `(s1,)`.

        If `get_prob` is set to True, this function returns a Symbol that will resolve to a list of
        outputs: `[ndarray_output, log_likelihood_output]`, where `log_likelihood_output` will resolve
        to the same shape as the sampled outputs in ndarray_output.
    """
    return _internal._sample_multinomial(data, shape, get_prob, dtype=dtype, **kwargs)


def shuffle(data, **kwargs):
    """Shuffle the elements randomly.

    This shuffles the array along the first axis.
    The order of the elements in each subarray does not change.
    For example, if a 2D array is given, the order of the rows randomly changes,
    but the order of the elements in each row does not change.

    Parameters
    ----------
    data : NDArray
        Input data array.

    Returns
    -------
    Symbol
        A new symbol representing the shuffled version of input `data`.

    Examples
    --------
    >>> data = mx.nd.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
    >>> a = mx.sym.Variable('a')
    >>> b = mx.sym.random.shuffle(a)
    >>> b.eval(a=data)
    [[ 0.  1.  2.]
     [ 6.  7.  8.]
     [ 3.  4.  5.]]
    <NDArray 2x3 @cpu(0)>
    >>> b.eval(a=data)
    [[ 3.  4.  5.]
     [ 0.  1.  2.]
     [ 6.  7.  8.]]
    <NDArray 2x3 @cpu(0)>
    """
    return _internal._shuffle(data, **kwargs)


def randint(low, high, shape=_Null, dtype=_Null, **kwargs):
    """Draw random samples from a discrete uniform distribution.

    Samples are uniformly distributed over the half-open interval *[low, high)*
    (includes *low*, but excludes *high*).

    Parameters
    ----------
    low : int, required
        Lower boundary of the output interval. All values generated will be
        greater than or equal to low.
    high : int, required
        Upper boundary of the output interval. All values generated will be
        less than high.
    shape : int or tuple of ints, optional
        The number of samples to draw. If shape is, e.g., `(m, n)` and `low` and
        `high` are scalars, output shape will be `(m, n)`.
    dtype : {'int32', 'int64'}, optional
        Data type of output samples. Default is 'int32'

    Returns
    -------
    Symbol
        If input `shape` has dimensions, e.g., `(m, n)`, and `low` and
        `high` are scalars, returned Symbol will resolve to shape `(m, n)`.
    """
    return _random_helper(_internal._random_randint, None,
                          [low, high], shape, dtype, kwargs)
