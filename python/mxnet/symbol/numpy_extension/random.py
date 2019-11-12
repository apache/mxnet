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

"""Namespace for operators used in Gluon dispatched by F=symbol."""

from __future__ import absolute_import
from ...context import current_context
from ..numpy import _internal as _npi

__all__ = ['bernoulli']


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
    out : _Symbol
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
    from ..numpy import _Symbol as np_symbol
    tensor_type_name = np_symbol
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
