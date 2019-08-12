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


"""I/O functions for ndarrays."""
from __future__ import absolute_import
import numpy as onp
from ..context import current_context
from .multiarray import array

__all__ = ['genfromtxt']


# TODO(junwu): Add doc
def genfromtxt(*args, **kwargs):
    """This is a wrapper of the official NumPy's `genfromtxt` function.
    Please refer to the documentation here
    https://docs.scipy.org/doc/numpy/reference/generated/numpy.genfromtxt.html.

    Notes
    -----
    This function has added an additional parameter `ctx` which allows to create
    ndarrays on the user-specified device.
    """
    ctx = kwargs.pop('ctx', current_context())
    if ctx is None:
        ctx = current_context()
    ret = onp.genfromtxt(*args, **kwargs)
    return array(ret, dtype=ret.dtype, ctx=ctx)
