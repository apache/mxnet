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

"""Namespace for operators used in Gluon dispatched by F=ndarray."""
from ...context import current_context
from ..numpy import _internal as _npi


__all__ = ['bernoulli', 'normal_n', 'uniform_n']


def bernoulli(prob, logit, size, dtype, ctx, out):
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
    from ...numpy import ndarray as np_ndarray
    tensor_type_name = np_ndarray
    if (prob is None) == (logit is None):
        raise ValueError(
            "Either `prob` or `logit` must be specified, but not both. " +
            "Received prob={}, logit={}".format(prob, logit))
    if dtype is None:
        dtype = 'float32'
    if ctx is None:
        ctx = current_context()
    if size == ():
        size = None
    if prob is not None:
        is_tensor = isinstance(prob, tensor_type_name)
        if is_tensor:
            return _npi.bernoulli(prob, prob=None, logit=None, is_logit=False,
                                  size=size, ctx=ctx, dtype=dtype, out=out)
        else:
            return _npi.bernoulli(prob=prob, logit=None, is_logit=False,
                                  size=size, ctx=ctx, dtype=dtype, out=out)
    else:
        is_tensor = isinstance(logit, tensor_type_name)
        if is_tensor:
            return _npi.bernoulli(logit, prob=None, logit=None, is_logit=True,
                                  size=size, ctx=ctx, dtype=dtype, out=out)
        else:
            return _npi.bernoulli(prob=None, logit=logit, is_logit=True,
                                  size=size, ctx=ctx, dtype=dtype, out=out)


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
    from ...numpy import ndarray as np_ndarray
    input_type = (isinstance(low, np_ndarray), isinstance(high, np_ndarray))
    if dtype is None:
        dtype = 'float32'
    if ctx is None:
        ctx = current_context()
    if batch_shape == ():
        batch_shape = None
    if input_type == (True, True):
        return _npi.uniform_n(low, high, low=None, high=None, size=batch_shape,
                              ctx=ctx, dtype=dtype)
    elif input_type == (False, True):
        return _npi.uniform_n(high, low=low, high=None, size=batch_shape,
                              ctx=ctx, dtype=dtype)
    elif input_type == (True, False):
        return _npi.uniform_n(low, low=None, high=high, size=batch_shape,
                              ctx=ctx, dtype=dtype)
    else:
        return _npi.uniform_n(low=low, high=high, size=batch_shape,
                              ctx=ctx, dtype=dtype)


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
    from ...numpy import ndarray as np_ndarray
    input_type = (isinstance(loc, np_ndarray), isinstance(scale, np_ndarray))
    if dtype is None:
        dtype = 'float32'
    if ctx is None:
        ctx = current_context()
    if batch_shape == ():
        batch_shape = None
    if input_type == (True, True):
        return _npi.normal_n(loc, scale, loc=None, scale=None, size=batch_shape,
                             ctx=ctx, dtype=dtype)
    elif input_type == (False, True):
        return _npi.normal_n(scale, loc=loc, scale=None, size=batch_shape,
                             ctx=ctx, dtype=dtype)
    elif input_type == (True, False):
        return _npi.normal_n(loc, loc=None, scale=scale, size=batch_shape,
                             ctx=ctx, dtype=dtype)
    else:
        return _npi.normal_n(loc=loc, scale=scale, size=batch_shape,
                             ctx=ctx, dtype=dtype)
