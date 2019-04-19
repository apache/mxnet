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

"""numpy namespace for operators used in Gluon APIs dispatched by F=ndarray module."""

from __future__ import absolute_import
import numpy as _np
from ..base import _sanity_check_params, use_np_compat
from ..context import current_context
from . import _internal

__all__ = ['zeros']


@use_np_compat
def zeros(shape, dtype=_np.float64, **kwargs):
    """Return a new array of given shape and type, filled with zeros.
    This function does not support the parameter `order` as in NumPy package.

    Parameters
    ----------
    shape : int or tuple of int
        The shape of the empty array.
    dtype : str or numpy.dtype, optional
        An optional value type (default is `float32`).
    ctx : Context, optional
        An optional device context (default is the current default context).

    Returns
    -------
    out : NDArray
        Array of zeros with the given shape, dtype, and ctx.
    """
    _sanity_check_params('zeros', ['order'], kwargs)
    ctx = kwargs.get('ctx', current_context())
    if ctx is None:
        ctx = current_context()
    dtype = _np.float64 if dtype is None else dtype
    return _internal._zeros(shape=shape, ctx=ctx, dtype=dtype, **kwargs).as_np_ndarray()
