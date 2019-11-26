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

from __future__ import absolute_import
from ..ndarray import numpy as _mx_nd_np

__all__ = ["randint", "uniform", "normal", "choice", "rand", "multinomial"]


def randint(low, high=None, size=None, dtype=None, ctx=None, out=None):
    r"""Return random integers from `low` (inclusive) to `high` (exclusive).

    Return random integers from the "discrete uniform" distribution of
    the specified dtype in the "half-open" interval [`low`, `high`). If
    `high` is None (the default), then results are from [0, `low`).

    Parameters
    ----------
    low : int
        Lowest (signed) integer to be drawn from the distribution (unless
        ``high=None``, in which case this parameter is one above the
        *highest* such integer).
    high : int, optional
        If provided, one above the largest (signed) integer to be drawn
        from the distribution (see above for behavior if ``high=None``).
    size : int or tuple of ints, optional
        Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
        ``m * n * k`` samples are drawn.  Default is None, in which case a
        single value is returned.
    dtype : dtype, optional
        Desired dtype of the result. All dtypes are determined by their
        name, i.e., 'int64', 'int', etc, so byteorder is not available
        and a specific precision may have different C types depending
        on the platform. The default value is 'np.int'.
    ctx : Context, optional
        Device context of output. Default is current context.
    out : ndarray, optional
        The output ndarray (default is `None`).

    Returns
    -------
    out : ndarray of ints
        `size`-shaped array of random integers from the appropriate
        distribution, or a single such random int if `size` not provided.

    Examples
    --------
    >>> np.random.randint(2, size=10)
    array([1, 0, 0, 0, 1, 1, 0, 0, 1, 0])
    >>> np.random.randint(1, size=10)
    array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

    Generate a 2 x 4 array of ints between 0 and 4, inclusive:

    >>> np.random.randint(5, size=(2, 4))
    array([[4, 0, 2, 1],
        [3, 2, 2, 0]])
    """
    return _mx_nd_np.random.randint(low, high, size, dtype, ctx, out)


def uniform(low=0.0, high=1.0, size=None, dtype=None, ctx=None, out=None):
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
    size : int or tuple of ints, optional
        Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
        ``m * n * k`` samples are drawn.  If size is ``None`` (default),
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
    return _mx_nd_np.random.uniform(low, high, size=size, ctx=ctx, dtype=dtype, out=out)


def normal(loc=0.0, scale=1.0, size=None, dtype=None, ctx=None, out=None):
    r"""Draw random samples from a normal (Gaussian) distribution.

    Samples are distributed according to a normal distribution parametrized
    by *loc* (mean) and *scale* (standard deviation).


    Parameters
    ----------
    loc : float, optional
        Mean (centre) of the distribution.
    scale : float, optional
        Standard deviation (spread or "width") of the distribution.
    size : int or tuple of ints, optional
        Output shape. If the given shape is, e.g., `(m, n, k)`, then `m * n * k`
        samples are drawn. If size is `None` (default), a scalar tensor containing
        a single value is returned if loc and scale are both scalars. Otherwise,
        ``np.broadcast(low, high).size`` samples are drawn.
    dtype : {'float16', 'float32', 'float64'}, optional
        Data type of output samples. Default is 'float32'
    ctx : Context, optional
        Device context of output, default is current context.
    out : ``ndarray``, optional
        Store output to an existing ``ndarray``.

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
    return _mx_nd_np.random.normal(loc, scale, size, dtype, ctx, out)


def multinomial(n, pvals, size=None, **kwargs):
    r"""
    Draw samples from a multinomial distribution.
    The multinomial distribution is a multivariate generalisation of the binomial distribution.
    Take an experiment with one of ``p`` possible outcomes. An example of such an experiment is throwing a dice,
    where the outcome can be 1 through 6. Each sample drawn from the distribution represents n such experiments.
    Its values, ``X_i = [X_0, X_1, ..., X_p]``, represent the number of times the outcome was ``i``.

    Parameters
    ----------
    n : int
        Number of experiments.
    pvals : sequence of floats, length p
        Probabilities of each of the p different outcomes. These should sum to 1.
    size : int or tuple of ints, optional
        Output shape. If the given shape is, e.g., ``(m, n, k)``, then ``m * n * k`` samples
        are drawn. Default is None, in which case a single value is returned.

    Returns
    -------
    out : ndarray
        The drawn samples, of shape size, if that was provided. If not, the shape is ``(N,)``.
        In other words, each entry ``out[i,j,...,:]`` is an N-dimensional value drawn from the distribution.

    Examples
    --------
    Throw a dice 1000 times, and 1000 times again:

    >>> np.random.multinomial(1000, [1/6.]*6, size=2)
    array([[164, 161, 179, 158, 150, 188],
           [178, 162, 177, 143, 163, 177]])

    A loaded die is more likely to land on number 6:

    >>> np.random.multinomial(100, [1/7.]*5 + [2/7.])
    array([19, 14, 12, 11, 21, 23])
    >>> np.random.multinomial(100, [1.0 / 3, 2.0 / 3])
    array([32, 68])
    """
    return _mx_nd_np.random.multinomial(n, pvals, size, **kwargs)


def choice(a, size=None, replace=True, p=None, ctx=None, out=None):
    r"""Generates a random sample from a given 1-D array

    Parameters
    -----------
    a : 1-D array-like or int
        If an ndarray, a random sample is generated from its elements.
        If an int, the random sample is generated as if a were np.arange(a)
    size : int or tuple of ints, optional
        Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
        ``m * n * k`` samples are drawn.  Default is None, in which case a
        single value is returned.
    replace : boolean, optional
        Whether the sample is with or without replacement
    p : 1-D array-like, optional
        The probabilities associated with each entry in a.
        If not given the sample assumes a uniform distribution over all
        entries in a.
    ctx : Context, optional
        Device context of output. Default is current context.

    Returns
    --------
    samples : ndarray
        The generated random samples

    Examples
    ---------
    Generate a uniform random sample from np.arange(5) of size 3:

    >>> np.random.choice(5, 3)
    array([0, 3, 4])
    >>> #This is equivalent to np.random.randint(0,5,3)

    Generate a non-uniform random sample from np.arange(5) of size 3:

    >>> np.random.choice(5, 3, p=[0.1, 0, 0.3, 0.6, 0])
    array([3, 3, 0])

    Generate a uniform random sample from np.arange(5) of size 3 without
    replacement:

    >>> np.random.choice(5, 3, replace=False)
    array([3,1,0])
    >>> #This is equivalent to np.random.permutation(np.arange(5))[:3]

    Generate a non-uniform random sample from np.arange(5) of size
    3 without replacement:

    >>> np.random.choice(5, 3, replace=False, p=[0.1, 0, 0.3, 0.6, 0])
    array([2, 3, 0])
    """
    return _mx_nd_np.random.choice(a, size, replace, p, ctx, out)


def rand(*size, **kwargs):
    r"""Random values in a given shape.

    Create an array of the given shape and populate it with random
    samples from a uniform distribution over [0, 1).
    Parameters
    ----------
    d0, d1, ..., dn : int, optional
        The dimensions of the returned array, should be all positive.
        If no argument is given a single Python float is returned.
    Returns
    -------
    out : ndarray
       Random values.
    Examples
    --------
    >>> np.random.rand(3,2)
    array([[ 0.14022471,  0.96360618],  #random
           [ 0.37601032,  0.25528411],  #random
           [ 0.49313049,  0.94909878]]) #random
    """
    output_shape = ()
    for s in size:
        output_shape += (s,)
    return _mx_nd_np.random.uniform(0, 1, size=output_shape, **kwargs)
