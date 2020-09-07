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

"""Namespace for ops used in imperative programming."""

from .. import random as _mx_rand
from ..ndarray import numpy_extension as _mx_nd_npx


__all__ = ['seed', 'bernoulli', 'normal_n', 'uniform_n']


def seed(seed, ctx='all'):  # pylint: disable=redefined-outer-name
    r"""Seeds the random number generators in MXNet.

    This affects the behavior of modules in MXNet that uses random number generators,
    like the dropout operator and `ndarray`'s random sampling operators.

    Parameters
    ----------
    seed : int
        The random number seed.

    ctx : Context
        The device context of the generator. The default is "all" which means seeding random
        number generators of all devices.

    Notes
    -----
    Random number generators in MXNet are device specific.
    `npx.random.seed(seed)` sets the state of each generator using `seed` and the
    device id. Therefore, random numbers generated from different devices can be different
    even if they are seeded using the same seed.

    To produce identical random number sequences independent of the device id,
    set optional `ctx` argument. This produces the same sequence of random numbers independent
    of the device id, but the sequence can be different on different kind of devices as MXNet's
    random number generators for CPU and GPU use different algorithms.

    Example
    -------
    >>> from mxnet import np, npx
    >>> npx.set_np()
    >>> npx.random.seed(0)
    >>> np.random.uniform()
    array(0.5488135)
    >>> npx.random.seed(128)
    >>> np.random.uniform()
    array(0.03812965)
    >>> npx.random.seed(128)
    >>> np.random.uniform()
    array(0.03812965)
    >>> npx.random.seed(128)
    >>> np.random.uniform(ctx=npx.gpu(0))
    array(0.9894903, ctx=gpu(0))
    >>> npx.random.seed(128)
    >>> np.random.uniform(ctx=npx.gpu(0))
    array(0.9894903, ctx=gpu(0))
    """
    _mx_rand.seed(seed_state=seed, ctx=ctx)


def bernoulli(prob=None, logit=None, size=None, dtype=None, ctx=None, out=None):
    """Creates a Bernoulli distribution parameterized by :attr:`prob`
    or :attr:`logit` (but not both).

    Samples are binary (0 or 1). They take the value `1` with probability `p`
    and `0` with probability `1 - p`.

    Parameters
    ----------
    prob : float, ndarray
        The probability of sampling '1'.
        Only one of prob or logit should be passed in.
    logit : float, ndarray
        The log-odds of sampling '1'.
        Only one of prob or logit should be passed in.
    size : int or tuple of ints, optional
        Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
        ``m * n * k`` samples are drawn.  Default is None, in which case a
        single value is returned.
    dtype : dtype, optional
        Desired dtype of the result. All dtypes are determined by their
        name, i.e., 'int64', 'int', etc, so byteorder is not available
        and a specific precision may have different C types depending
        on the platform. The default value is 'np.float32'.
    ctx : Context, optional
        Device context of output. Default is current context.
    out : symbol, optional
        The output symbol (default is `None`).

    Returns
    -------
    out : ndarray
        Drawn samples from the parameterized bernoulli distribution.

    Examples
    --------
    >>> prob = np.random.uniform(size=(4,4))
    >>> logit = np.log(prob) - np.log(1 - prob)
    >>> npx.random.bernoulli(logit=logit)
    array([[0., 1., 1., 1.],
        [0., 1., 1., 1.],
        [0., 1., 0., 0.],
        [1., 0., 1., 0.]])

    >>> npx.random.bernoulli(prob=prob)
    array([[0., 1., 0., 1.],
        [1., 1., 1., 1.],
        [1., 1., 1., 0.],
        [1., 0., 1., 0.]])
    """
    return _mx_nd_npx.random.bernoulli(prob, logit, size, dtype, ctx, out)


def uniform_n(low=0.0, high=1.0, batch_shape=None, dtype=None, ctx=None):
    r"""Draw samples from a uniform distribution.

    Samples are uniformly distributed over the half-open interval
    ``[low, high)`` (includes low, but excludes high).  In other words,
    any value within the given interval is equally likely to be drawn
    by `uniform`.

    Parameters
    ----------
    low : float, ndarray, optional
        Lower boundary of the output interval.  All values generated will be
        greater than or equal to low.  The default value is 0.
    high : float, ndarray, optional
        Upper boundary of the output interval.  All values generated will be
        less than high.  The default value is 1.0.
    batch_shape : int or tuple of ints, optional
        Batch shape.  If the given shape is, e.g., ``(m, n, k)``, then
        ``m * n * k * broadcast(low, high).size`` samples are drawn.
        If size is ``None`` (default),
        a scalar tensor containing a single value is returned if
        ``low`` and ``high`` are both scalars. Otherwise,
        ``np.broadcast(low, high).size`` samples are drawn.
    dtype : {'float16', 'float32', 'float64'}, optional
        Data type of output samples. Default is 'float32'
    ctx : Context, optional
        Device context of output. Default is current context.

    Returns
    -------
    out : ndarray
        Drawn samples from the parameterized uniform distribution.

    See Also
    --------
    randint : Discrete uniform distribution, yielding integers.
    rand : Convenience function that accepts dimensions as input, e.g.,
           ``rand(2,2)`` would generate a 2-by-2 array of floats,
           uniformly distributed over ``[0, 1)``.

    Notes
    -----
    The probability density function of the uniform distribution is

    .. math:: p(x) = \frac{1}{b - a}

    anywhere within the interval ``[a, b)``, and zero elsewhere.

    When ``high`` == ``low``, values of ``low`` will be returned.
    If ``high`` < ``low``, the results are officially undefined
    and may eventually raise an error, i.e. do not rely on this
    function to behave when passed arguments satisfying that
    inequality condition.
    """
    return _mx_nd_npx.random.uniform_n(low, high, batch_shape=batch_shape, ctx=ctx, dtype=dtype)


def normal_n(loc=0.0, scale=1.0, batch_shape=None, dtype=None, ctx=None):
    r"""Draw random samples from a normal (Gaussian) distribution.

    Samples are distributed according to a normal distribution parametrized
    by *loc* (mean) and *scale* (standard deviation).


    Parameters
    ----------
    loc : float, optional
        Mean (centre) of the distribution.
    scale : float, optional
        Standard deviation (spread or "width") of the distribution.
    batch_shape : int or tuple of ints, optional
        Batch shape.  If the given shape is, e.g., ``(m, n, k)``, then
        ``m * n * k * broadcast(low, high).size`` samples are drawn.
        If size is ``None`` (default),
        a scalar tensor containing a single value is returned if
        ``low`` and ``high`` are both scalars. Otherwise,
        ``np.broadcast(loc, scale).size`` samples are drawn.
    dtype : {'float16', 'float32', 'float64'}, optional
        Data type of output samples. Default is 'float32'
    ctx : Context, optional
        Device context of output, default is current context.

    Returns
    -------
    out : ndarray
        Drawn samples from the parameterized normal distribution.

    Notes
    -----
    The probability density for the Gaussian distribution is

    .. math:: p(x) = \frac{1}{\sqrt{ 2 \pi \sigma^2 }}
                     e^{ - \frac{ (x - \mu)^2 } {2 \sigma^2} },

    where :math:`\mu` is the mean and :math:`\sigma` the standard
    deviation. The square of the standard deviation, :math:`\sigma^2`,
    is called the variance.

    The function has its peak at the mean, and its "spread" increases with
    the standard deviation (the function reaches 0.607 times its maximum at
    :math:`x + \sigma` and :math:`x - \sigma` [2]_).  This implies that
    `numpy.random.normal` is more likely to return samples lying close to
    the mean, rather than those far away.

    References
    ----------
    .. [1] Wikipedia, "Normal distribution",
           https://en.wikipedia.org/wiki/Normal_distribution
    .. [2] P. R. Peebles Jr., "Central Limit Theorem" in "Probability,
           Random Variables and Random Signal Principles", 4th ed., 2001,
           pp. 51, 51, 125.

    Examples
    --------
    >>> mu, sigma = 0, 0.1 # mean and standard deviation
    >>> s = np.random.normal(mu, sigma, 1000)

    Verify the mean and the variance:

    >>> np.abs(mu - np.mean(s)) < 0.01
    array(True)
    """
    return _mx_nd_npx.random.normal_n(loc, scale, batch_shape, dtype, ctx)
