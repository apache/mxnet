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

"""Random distribution generator NDArray API of MXNet."""

from ..base import numeric_types, _Null
from ..context import current_context
from . import _internal
from .ndarray import NDArray


__all__ = ['uniform', 'normal', 'randn', 'poisson', 'exponential', 'gamma',
           'multinomial', 'negative_binomial', 'generalized_negative_binomial',
           'shuffle', 'randint']


def _random_helper(random, sampler, params, shape, dtype, ctx, out, kwargs):
    """Helper function for random generators."""
    if isinstance(params[0], NDArray):
        for i in params[1:]:
            assert isinstance(i, NDArray), \
                "Distribution parameters must all have the same type, but got " \
                "both %s and %s."%(type(params[0]), type(i))
        return sampler(*params, shape=shape, dtype=dtype, out=out, **kwargs)
    elif isinstance(params[0], numeric_types):
        if ctx is None:
            ctx = current_context()
        if shape is _Null and out is None:
            shape = 1
        for i in params[1:]:
            assert isinstance(i, numeric_types), \
                "Distribution parameters must all have the same type, but got " \
                "both %s and %s."%(type(params[0]), type(i))
        return random(*params, shape=shape, dtype=dtype, ctx=ctx, out=out, **kwargs)

    raise ValueError("Distribution parameters must be either NDArray or numbers, "
                     "but got %s."%type(params[0]))


def uniform(low=0, high=1, shape=_Null, dtype=_Null, ctx=None, out=None, **kwargs):
    """Draw random samples from a uniform distribution.

    Samples are uniformly distributed over the half-open interval *[low, high)*
    (includes *low*, but excludes *high*).

    Parameters
    ----------
    low : float or NDArray, optional
        Lower boundary of the output interval. All values generated will be
        greater than or equal to low. The default value is 0.
    high : float or NDArray, optional
        Upper boundary of the output interval. All values generated will be
        less than high. The default value is 1.0.
    shape : int or tuple of ints, optional
        The number of samples to draw. If shape is, e.g., `(m, n)` and `low` and
        `high` are scalars, output shape will be `(m, n)`. If `low` and `high`
        are NDArrays with shape, e.g., `(x, y)`, then output will have shape
        `(x, y, m, n)`, where `m*n` samples are drawn for each `[low, high)` pair.
    dtype : {'float16', 'float32', 'float64'}, optional
        Data type of output samples. Default is 'float32'
    ctx : Context, optional
        Device context of output. Default is current context. Overridden by
        `low.context` when `low` is an NDArray.
    out : NDArray, optional
        Store output to an existing NDArray.

    Returns
    -------
    NDArray
        An NDArray of type `dtype`. If input `shape` has shape, e.g.,
        `(m, n)` and `low` and `high` are scalars, output shape will be `(m, n)`.
        If `low` and `high` are NDArrays with shape, e.g., `(x, y)`, then the
        return NDArray will have shape `(x, y, m, n)`, where `m*n` uniformly distributed
        samples are drawn for each `[low, high)` pair.

    Examples
    --------
    >>> mx.nd.random.uniform(0, 1)
    [ 0.54881352]
    <NDArray 1 @cpu(0)
    >>> mx.nd.random.uniform(0, 1, ctx=mx.gpu(0))
    [ 0.92514056]
    <NDArray 1 @gpu(0)>
    >>> mx.nd.random.uniform(-1, 1, shape=(2,))
    [ 0.71589124  0.08976638]
    <NDArray 2 @cpu(0)>
    >>> low = mx.nd.array([1,2,3])
    >>> high = mx.nd.array([2,3,4])
    >>> mx.nd.random.uniform(low, high, shape=2)
    [[ 1.78653979  1.93707538]
     [ 2.01311183  2.37081361]
     [ 3.30491424  3.69977832]]
    <NDArray 3x2 @cpu(0)>
    """
    return _random_helper(_internal._random_uniform, _internal._sample_uniform,
                          [low, high], shape, dtype, ctx, out, kwargs)


def normal(loc=0, scale=1, shape=_Null, dtype=_Null, ctx=None, out=None, **kwargs):
    """Draw random samples from a normal (Gaussian) distribution.

    Samples are distributed according to a normal distribution parametrized
    by *loc* (mean) and *scale* (standard deviation).


    Parameters
    ----------
    loc : float or NDArray, optional
        Mean (centre) of the distribution.
    scale : float or NDArray, optional
        Standard deviation (spread or width) of the distribution.
    shape : int or tuple of ints, optional
        The number of samples to draw. If shape is, e.g., `(m, n)` and `loc` and
        `scale` are scalars, output shape will be `(m, n)`. If `loc` and `scale`
        are NDArrays with shape, e.g., `(x, y)`, then output will have shape
        `(x, y, m, n)`, where `m*n` samples are drawn for each `[loc, scale)` pair.
    dtype : {'float16', 'float32', 'float64'}, optional
        Data type of output samples. Default is 'float32'
    ctx : Context, optional
        Device context of output. Default is current context. Overridden by
        `loc.context` when `loc` is an NDArray.
    out : NDArray, optional
        Store output to an existing NDArray.

    Returns
    -------
    NDArray
        An NDArray of type `dtype`. If input `shape` has shape, e.g., `(m, n)` and
        `loc` and `scale` are scalars, output shape will be `(m, n)`. If `loc` and
        `scale` are NDArrays with shape, e.g., `(x, y)`, then output will have shape
        `(x, y, m, n)`, where `m*n` samples are drawn for each `[loc, scale)` pair.

    Examples
    --------
    >>> mx.nd.random.normal(0, 1)
    [ 2.21220636]
    <NDArray 1 @cpu(0)>
    >>> mx.nd.random.normal(0, 1, ctx=mx.gpu(0))
    [ 0.29253659]
    <NDArray 1 @gpu(0)>
    >>> mx.nd.random.normal(-1, 1, shape=(2,))
    [-0.2259962  -0.51619542]
    <NDArray 2 @cpu(0)>
    >>> loc = mx.nd.array([1,2,3])
    >>> scale = mx.nd.array([2,3,4])
    >>> mx.nd.random.normal(loc, scale, shape=2)
    [[ 0.55912292  3.19566321]
     [ 1.91728961  2.47706747]
     [ 2.79666662  5.44254589]]
    <NDArray 3x2 @cpu(0)>
    """
    return _random_helper(_internal._random_normal, _internal._sample_normal,
                          [loc, scale], shape, dtype, ctx, out, kwargs)


def randn(*shape, **kwargs):
    """Draw random samples from a normal (Gaussian) distribution.

    Samples are distributed according to a normal distribution parametrized
    by *loc* (mean) and *scale* (standard deviation).


    Parameters
    ----------
    loc : float or NDArray
        Mean (centre) of the distribution.
    scale : float or NDArray
        Standard deviation (spread or width) of the distribution.
    shape : int or tuple of ints
        The number of samples to draw. If shape is, e.g., `(m, n)` and `loc` and
        `scale` are scalars, output shape will be `(m, n)`. If `loc` and `scale`
        are NDArrays with shape, e.g., `(x, y)`, then output will have shape
        `(x, y, m, n)`, where `m*n` samples are drawn for each `[loc, scale)` pair.
    dtype : {'float16', 'float32', 'float64'}
        Data type of output samples. Default is 'float32'
    ctx : Context
        Device context of output. Default is current context. Overridden by
        `loc.context` when `loc` is an NDArray.
    out : NDArray
        Store output to an existing NDArray.

    Returns
    -------
    NDArray
        If input `shape` has shape, e.g., `(m, n)` and `loc` and `scale` are scalars, output
        shape will be `(m, n)`. If `loc` and `scale` are NDArrays with shape, e.g., `(x, y)`,
        then output will have shape `(x, y, m, n)`, where `m*n` samples are drawn for
        each `[loc, scale)` pair.

    Examples
    --------
    >>> mx.nd.random.randn()
    2.21220636
    <NDArray 1 @cpu(0)>
    >>> mx.nd.random.randn(2, 2)
    [[-1.856082   -1.9768796 ]
    [-0.20801921  0.2444218 ]]
    <NDArray 2x2 @cpu(0)>
    >>> mx.nd.random.randn(2, 3, loc=5, scale=1)
    [[4.19962   4.8311777 5.936328 ]
    [5.357444  5.7793283 3.9896927]]
    <NDArray 2x3 @cpu(0)>
    """
    loc = kwargs.pop('loc', 0)
    scale = kwargs.pop('scale', 1)
    dtype = kwargs.pop('dtype', _Null)
    ctx = kwargs.pop('ctx', None)
    out = kwargs.pop('out', None)
    assert isinstance(loc, (int, float, NDArray))
    assert isinstance(scale, (int, float, NDArray))
    return _random_helper(_internal._random_normal, _internal._sample_normal,
                          [loc, scale], shape, dtype, ctx, out, kwargs)


def poisson(lam=1, shape=_Null, dtype=_Null, ctx=None, out=None, **kwargs):
    """Draw random samples from a Poisson distribution.

    Samples are distributed according to a Poisson distribution parametrized
    by *lambda* (rate). Samples will always be returned as a floating point data type.

    Parameters
    ----------
    lam : float or NDArray, optional
        Expectation of interval, should be >= 0.
    shape : int or tuple of ints, optional
        The number of samples to draw. If shape is, e.g., `(m, n)` and `lam` is
        a scalar, output shape will be `(m, n)`. If `lam`
        is an NDArray with shape, e.g., `(x, y)`, then output will have shape
        `(x, y, m, n)`, where `m*n` samples are drawn for each entry in `lam`.
    dtype : {'float16', 'float32', 'float64'}, optional
        Data type of output samples. Default is 'float32'
    ctx : Context, optional
        Device context of output. Default is current context. Overridden by
        `lam.context` when `lam` is an NDArray.
    out : NDArray, optional
        Store output to an existing NDArray.

    Returns
    -------
    NDArray
        If input `shape` has shape, e.g., `(m, n)` and `lam` is
        a scalar, output shape will be `(m, n)`. If `lam`
        is an NDArray with shape, e.g., `(x, y)`, then output will have shape
        `(x, y, m, n)`, where `m*n` samples are drawn for each entry in `lam`.

    Examples
    --------
    >>> mx.nd.random.poisson(1)
    [ 1.]
    <NDArray 1 @cpu(0)>
    >>> mx.nd.random.poisson(1, shape=(2,))
    [ 0.  2.]
    <NDArray 2 @cpu(0)>
    >>> lam = mx.nd.array([1,2,3])
    >>> mx.nd.random.poisson(lam, shape=2)
    [[ 1.  3.]
     [ 3.  2.]
     [ 2.  3.]]
    <NDArray 3x2 @cpu(0)>
    """
    return _random_helper(_internal._random_poisson, _internal._sample_poisson,
                          [lam], shape, dtype, ctx, out, kwargs)


def exponential(scale=1, shape=_Null, dtype=_Null, ctx=None, out=None, **kwargs):
    r"""Draw samples from an exponential distribution.

    Its probability density function is

    .. math:: f(x; \frac{1}{\beta}) = \frac{1}{\beta} \exp(-\frac{x}{\beta}),

    for x > 0 and 0 elsewhere. \beta is the scale parameter, which is the
    inverse of the rate parameter \lambda = 1/\beta.

    Parameters
    ----------
    scale : float or NDArray, optional
        The scale parameter, \beta = 1/\lambda.
    shape : int or tuple of ints, optional
        The number of samples to draw. If shape is, e.g., `(m, n)` and `scale` is
        a scalar, output shape will be `(m, n)`. If `scale`
        is an NDArray with shape, e.g., `(x, y)`, then output will have shape
        `(x, y, m, n)`, where `m*n` samples are drawn for each entry in `scale`.
    dtype : {'float16', 'float32', 'float64'}, optional
        Data type of output samples. Default is 'float32'
    ctx : Context, optional
        Device context of output. Default is current context. Overridden by
        `scale.context` when `scale` is an NDArray.
    out : NDArray, optional
        Store output to an existing NDArray.

    Returns
    -------
    NDArray
        If input `shape` has shape, e.g., `(m, n)` and `scale` is a scalar, output shape will
        be `(m, n)`. If `scale` is an NDArray with shape, e.g., `(x, y)`, then `output`
        will have shape `(x, y, m, n)`, where `m*n` samples are drawn for each entry in scale.

    Examples
    --------
    >>> mx.nd.random.exponential(1)
    [ 0.79587454]
    <NDArray 1 @cpu(0)>
    >>> mx.nd.random.exponential(1, shape=(2,))
    [ 0.89856035  1.25593066]
    <NDArray 2 @cpu(0)>
    >>> scale = mx.nd.array([1,2,3])
    >>> mx.nd.random.exponential(scale, shape=2)
    [[  0.41063145   0.42140478]
     [  2.59407091  10.12439728]
     [  2.42544937   1.14260709]]
    <NDArray 3x2 @cpu(0)>
    """
    return _random_helper(_internal._random_exponential, _internal._sample_exponential,
                          [1.0/scale], shape, dtype, ctx, out, kwargs)


def gamma(alpha=1, beta=1, shape=_Null, dtype=_Null, ctx=None, out=None, **kwargs):
    """Draw random samples from a gamma distribution.

    Samples are distributed according to a gamma distribution parametrized
    by *alpha* (shape) and *beta* (scale).

    Parameters
    ----------
    alpha : float or NDArray, optional
        The shape of the gamma distribution. Should be greater than zero.
    beta : float or NDArray, optional
        The scale of the gamma distribution. Should be greater than zero.
        Default is equal to 1.
    shape : int or tuple of ints, optional
        The number of samples to draw. If shape is, e.g., `(m, n)` and `alpha` and
        `beta` are scalars, output shape will be `(m, n)`. If `alpha` and `beta`
        are NDArrays with shape, e.g., `(x, y)`, then output will have shape
        `(x, y, m, n)`, where `m*n` samples are drawn for each `[alpha, beta)` pair.
    dtype : {'float16', 'float32', 'float64'}, optional
        Data type of output samples. Default is 'float32'
    ctx : Context, optional
        Device context of output. Default is current context. Overridden by
        `alpha.context` when `alpha` is an NDArray.
    out : NDArray, optional
        Store output to an existing NDArray.

    Returns
    -------
    NDArray
        If input `shape` has shape, e.g., `(m, n)` and `alpha` and `beta` are scalars, output
        shape will be `(m, n)`. If `alpha` and `beta` are NDArrays with shape, e.g.,
        `(x, y)`, then output will have shape `(x, y, m, n)`, where `m*n` samples are
        drawn for each `[alpha, beta)` pair.

    Examples
    --------
    >>> mx.nd.random.gamma(1, 1)
    [ 1.93308783]
    <NDArray 1 @cpu(0)>
    >>> mx.nd.random.gamma(1, 1, shape=(2,))
    [ 0.48216391  2.09890771]
    <NDArray 2 @cpu(0)>
    >>> alpha = mx.nd.array([1,2,3])
    >>> beta = mx.nd.array([2,3,4])
    >>> mx.nd.random.gamma(alpha, beta, shape=2)
    [[  3.24343276   0.94137681]
     [  3.52734375   0.45568955]
     [ 14.26264095  14.0170126 ]]
    <NDArray 3x2 @cpu(0)>
    """
    return _random_helper(_internal._random_gamma, _internal._sample_gamma,
                          [alpha, beta], shape, dtype, ctx, out, kwargs)


def negative_binomial(k=1, p=1, shape=_Null, dtype=_Null, ctx=None,
                      out=None, **kwargs):
    """Draw random samples from a negative binomial distribution.

    Samples are distributed according to a negative binomial distribution
    parametrized by *k* (limit of unsuccessful experiments) and *p* (failure
    probability in each experiment). Samples will always be returned as a
    floating point data type.

    Parameters
    ----------
    k : float or NDArray, optional
        Limit of unsuccessful experiments, > 0.
    p : float or NDArray, optional
        Failure probability in each experiment, >= 0 and <=1.
    shape : int or tuple of ints, optional
        The number of samples to draw. If shape is, e.g., `(m, n)` and `k` and
        `p` are scalars, output shape will be `(m, n)`. If `k` and `p`
        are NDArrays with shape, e.g., `(x, y)`, then output will have shape
        `(x, y, m, n)`, where `m*n` samples are drawn for each `[k, p)` pair.
    dtype : {'float16', 'float32', 'float64'}, optional
        Data type of output samples. Default is 'float32'
    ctx : Context, optional
        Device context of output. Default is current context. Overridden by
        `k.context` when `k` is an NDArray.
    out : NDArray, optional
        Store output to an existing NDArray.

    Returns
    -------
    NDArray
        If input `shape` has shape, e.g., `(m, n)` and `k` and `p` are scalars, output shape
        will be `(m, n)`. If `k` and `p` are NDArrays with shape, e.g., `(x, y)`, then
        output will have shape `(x, y, m, n)`, where `m*n` samples are drawn for each `[k, p)` pair.

    Examples
    --------
    >>> mx.nd.random.negative_binomial(10, 0.5)
    [ 4.]
    <NDArray 1 @cpu(0)>
    >>> mx.nd.random.negative_binomial(10, 0.5, shape=(2,))
    [ 3.  4.]
    <NDArray 2 @cpu(0)>
    >>> k = mx.nd.array([1,2,3])
    >>> p = mx.nd.array([0.2,0.4,0.6])
    >>> mx.nd.random.negative_binomial(k, p, shape=2)
    [[ 3.  2.]
     [ 4.  4.]
     [ 0.  5.]]
    <NDArray 3x2 @cpu(0)>
    """
    return _random_helper(_internal._random_negative_binomial,
                          _internal._sample_negative_binomial,
                          [k, p], shape, dtype, ctx, out, kwargs)


def generalized_negative_binomial(mu=1, alpha=1, shape=_Null, dtype=_Null, ctx=None,
                                  out=None, **kwargs):
    """Draw random samples from a generalized negative binomial distribution.

    Samples are distributed according to a generalized negative binomial
    distribution parametrized by *mu* (mean) and *alpha* (dispersion).
    *alpha* is defined as *1/k* where *k* is the failure limit of the
    number of unsuccessful experiments (generalized to real numbers).
    Samples will always be returned as a floating point data type.

    Parameters
    ----------
    mu : float or NDArray, optional
        Mean of the negative binomial distribution.
    alpha : float or NDArray, optional
        Alpha (dispersion) parameter of the negative binomial distribution.
    shape : int or tuple of ints, optional
        The number of samples to draw. If shape is, e.g., `(m, n)` and `mu` and
        `alpha` are scalars, output shape will be `(m, n)`. If `mu` and `alpha`
        are NDArrays with shape, e.g., `(x, y)`, then output will have shape
        `(x, y, m, n)`, where `m*n` samples are drawn for each `[mu, alpha)` pair.
    dtype : {'float16', 'float32', 'float64'}, optional
        Data type of output samples. Default is 'float32'
    ctx : Context, optional
        Device context of output. Default is current context. Overridden by
        `mu.context` when `mu` is an NDArray.
    out : NDArray, optional
        Store output to an existing NDArray.

    Returns
    -------
    NDArray
        If input `shape` has shape, e.g., `(m, n)` and `mu` and `alpha` are scalars, output
        shape will be `(m, n)`. If `mu` and `alpha` are NDArrays with shape, e.g., `(x, y)`,
        then output will have shape `(x, y, m, n)`, where `m*n` samples are drawn for
        each `[mu, alpha)` pair.

    Examples
    --------
    >>> mx.nd.random.generalized_negative_binomial(10, 0.5)
    [ 19.]
    <NDArray 1 @cpu(0)>
    >>> mx.nd.random.generalized_negative_binomial(10, 0.5, shape=(2,))
    [ 30.  21.]
    <NDArray 2 @cpu(0)>
    >>> mu = mx.nd.array([1,2,3])
    >>> alpha = mx.nd.array([0.2,0.4,0.6])
    >>> mx.nd.random.generalized_negative_binomial(mu, alpha, shape=2)
    [[ 4.  0.]
     [ 3.  2.]
     [ 6.  2.]]
    <NDArray 3x2 @cpu(0)>
    """
    return _random_helper(_internal._random_generalized_negative_binomial,
                          _internal._sample_generalized_negative_binomial,
                          [mu, alpha], shape, dtype, ctx, out, kwargs)


def multinomial(data, shape=_Null, get_prob=False, out=None, dtype='int32', **kwargs):
    """Concurrent sampling from multiple multinomial distributions.

    .. note:: The input distribution must be normalized, i.e. `data` must sum to
              1 along its last dimension.

    Parameters
    ----------
    data : NDArray
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
    out : NDArray, optional
        Store output to an existing NDArray.
    dtype : str or numpy.dtype, optional
        Data type of the sample output array. The default is int32.
        Note that the data type of the log likelihood array is the same with that of `data`.

    Returns
    -------
    List, or NDArray
        For input `data` with `n` dimensions and shape `(d1, d2, ..., dn-1, k)`, and input
        `shape` with shape `(s1, s2, ..., sx)`, returns an NDArray with shape
        `(d1, d2, ... dn-1, s1, s2, ..., sx)`. The `s1, s2, ... sx` dimensions of the
        returned NDArray consist of 0-indexed values sampled from each respective multinomial
        distribution provided in the `k` dimension of `data`.

        For the case `n`=1, and `x`=1 (one shape dimension), returned NDArray has shape `(s1,)`.

        If `get_prob` is set to True, this function returns a list of format:
        `[ndarray_output, log_likelihood_output]`, where `log_likelihood_output` is an NDArray of the
        same shape as the sampled outputs.

    Examples
    --------
    >>> probs = mx.nd.array([0, 0.1, 0.2, 0.3, 0.4])
    >>> mx.nd.random.multinomial(probs)
    [3]
    <NDArray 1 @cpu(0)>
    >>> probs = mx.nd.array([[0, 0.1, 0.2, 0.3, 0.4], [0.4, 0.3, 0.2, 0.1, 0]])
    >>> mx.nd.random.multinomial(probs)
    [3 1]
    <NDArray 2 @cpu(0)>
    >>> mx.nd.random.multinomial(probs, shape=2)
    [[4 4]
     [1 2]]
    <NDArray 2x2 @cpu(0)>
    >>> mx.nd.random.multinomial(probs, get_prob=True)
    [3 2]
    <NDArray 2 @cpu(0)>
    [-1.20397282 -1.60943794]
    <NDArray 2 @cpu(0)>
    """
    return _internal._sample_multinomial(data, shape, get_prob, out=out, dtype=dtype, **kwargs)


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
    out : NDArray, optional
        Array to store the result.

    Returns
    -------
    NDArray
        A new NDArray with the same shape and type as input `data`, but
        with items in the first axis of the returned NDArray shuffled randomly.
        The original input `data` is not modified.

    Examples
    --------
    >>> data = mx.nd.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
    >>> mx.nd.random.shuffle(data)
    [[ 0.  1.  2.]
     [ 6.  7.  8.]
     [ 3.  4.  5.]]
    <NDArray 2x3 @cpu(0)>
    >>> mx.nd.random.shuffle(data)
    [[ 3.  4.  5.]
     [ 0.  1.  2.]
     [ 6.  7.  8.]]
    <NDArray 2x3 @cpu(0)>
    """
    return _internal._shuffle(data, **kwargs)


def randint(low, high, shape=_Null, dtype=_Null, ctx=None, out=None, **kwargs):
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
    ctx : Context, optional
        Device context of output. Default is current context. Overridden by
        `low.context` when `low` is an NDArray.
    out : NDArray, optional
        Store output to an existing NDArray.

    Returns
    -------
    NDArray
        An NDArray of type `dtype`. If input `shape` has shape, e.g.,
        `(m, n)`, the returned NDArray will shape will be `(m, n)`. Contents
        of the returned NDArray will be samples from the interval `[low, high)`.

    Examples
    --------
    >>> mx.nd.random.randint(5, 100)
    [ 90]
    <NDArray 1 @cpu(0)
    >>> mx.nd.random.randint(-10, 2, ctx=mx.gpu(0))
    [ -8]
    <NDArray 1 @gpu(0)>
    >>> mx.nd.random.randint(-10, 10, shape=(2,))
    [ -5  4]
    <NDArray 2 @cpu(0)>
    """
    return _random_helper(_internal._random_randint, None,
                          [low, high], shape, dtype, ctx, out, kwargs)
