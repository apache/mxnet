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

__all__ = ['uniform', 'normal']


def uniform(low=0.0, high=1.0, size=None, **kwargs):
    """Draw samples from a uniform distribution.

    Samples are uniformly distributed over the half-open interval
    ``[low, high)`` (includes low, but excludes high).  In other words,
    any value within the given interval is equally likely to be drawn
    by `uniform`.

    Parameters
    ----------
    low : float, optional
        Lower boundary of the output interval.  All values generated will be
        greater than or equal to low.  The default value is 0.
    high : float
        Upper boundary of the output interval.  All values generated will be
        less than high.  The default value is 1.0.
    size : int or tuple of ints, optional
        Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
        ``m * n * k`` samples are drawn.  If size is ``None`` (default),
        a scalar tensor containing a single value is returned if
        ``low`` and ``high`` are both scalars.
    dtype : {'float16', 'float32', 'float64'}, optional
        Data type of output samples. Default is 'float32'
    ctx : Context, optional
        Device context of output. Default is current context.
    out : ndarray, optional
        Store output to an existing ndarray.

    Returns
    -------
    out : ndarray
        Drawn samples from the parameterized uniform distribution.


    Notes
    -----
    This function currently does not support ``low`` and ``high`` as ndarrays.
    """
    return _mx_nd_np.random.uniform(low, high, size, **kwargs)


def normal(loc=0.0, scale=1.0, size=None, **kwargs):
    """Draw random samples from a normal (Gaussian) distribution.

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
        a single value is returned if loc and scale are both scalars.
    dtype : {'float16', 'float32', 'float64'}, optional
        Data type of output samples. Default is 'float32'
    ctx : Context, optional
        Device context of output. Default is current context.
    out : ``ndarray``, optional
        Store output to an existing ``ndarray``.

    Returns
    -------
    out : ndarray
        Drawn samples from the parameterized normal distribution.

    Notes
    -----
    This function currently does not support ``loc`` and ``scale`` as ndarrays.
    """
    return _mx_nd_np.random.normal(loc, scale, size, **kwargs)


def multinomial(n, pvals, size=None, **kwargs):
    """multinomial(n, pvals, size=None)

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
