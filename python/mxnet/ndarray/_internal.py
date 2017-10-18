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

# pylint: disable=wildcard-import, unused-import
"""NDArray namespace used to register internal functions."""
import os as _os
import sys as _sys

import numpy as np

try:
    if int(_os.environ.get("MXNET_ENABLE_CYTHON", True)) == 0:
        from .._ctypes.ndarray import NDArrayBase, CachedOp
        from .._ctypes.ndarray import _set_ndarray_class, _imperative_invoke
    elif _sys.version_info >= (3, 0):
        from .._cy3.ndarray import NDArrayBase, CachedOp
        from .._cy3.ndarray import _set_ndarray_class, _imperative_invoke
    else:
        from .._cy2.ndarray import NDArrayBase, CachedOp
        from .._cy2.ndarray import _set_ndarray_class, _imperative_invoke
except ImportError:
    if int(_os.environ.get("MXNET_ENFORCE_CYTHON", False)) != 0:
        raise ImportError("Cython Module cannot be loaded but MXNET_ENFORCE_CYTHON=1")
    from .._ctypes.ndarray import NDArrayBase, CachedOp
    from .._ctypes.ndarray import _set_ndarray_class, _imperative_invoke

from ..base import _Null
try:
    from .gen__internal import * # pylint: disable=unused-wildcard-import
except ImportError:
    pass

__all__ = ['NDArrayBase', 'CachedOp', '_imperative_invoke', '_set_ndarray_class']
