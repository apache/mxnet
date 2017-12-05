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

# coding: utf-8
# pylint: disable=wildcard-import, unused-wildcard-import, too-many-lines

"""MKLDNN NDArray API of MXNet."""

from __future__ import absolute_import
from __future__ import division
try:
    from __builtin__ import slice as py_slice
    from __builtin__ import sum as py_sum
except ImportError:
    from builtins import slice as py_slice
    from builtins import sum as py_sum

import ctypes
import warnings

__all__ = ["_ndarray_cls", "MKLNDArray"]

import numpy as np
from ..base import _LIB, numeric_types
from ..base import c_array, mx_real_t, integer_types
from ..base import mx_uint, NDArrayHandle, check_call
from ..context import Context
from . import _internal
from . import op
from ._internal import _set_ndarray_class
from .ndarray import NDArray, _storage_type, _DTYPE_NP_TO_MX, _DTYPE_MX_TO_NP
from .ndarray import _STORAGE_TYPE_STR_TO_ID, _STORAGE_TYPE_MKLDNN
from .ndarray import _STORAGE_TYPE_UNDEFINED, _STORAGE_TYPE_DEFAULT
from .ndarray import zeros as _zeros_ndarray
from .ndarray import array as _array

class MKLNDArray(NDArray):
    """The base class of an NDArray stored in a MKLDNN storage format.
    """

    def __repr__(self):
        """Returns a string representation of the sparse array."""
        shape_info = 'x'.join(['%d' % x for x in self.shape])
        # The data content is not displayed since the array usually has big shape
        return '\n<%s %s @%s>' % (self.__class__.__name__,
                                  shape_info, self.context)

    # TODO
    def _at(self, idx):
        raise NotSupportedForMKLNDArray(self._at, '[idx]', idx)

    def _slice(self, start, stop):
        return op.slice(self, begin=start, end=stop)

    # TODO
    def astype(self, dtype):
        """Returns a copy of the array after casting to a specified type.
        Parameters
        ----------
        dtype : numpy.dtype or str
            The type of the returned array.
        Examples
        --------
        >>> x = mx.nd.sparse.zeros('row_sparse', (2,3), dtype='float32')
        >>> y = x.astype('int32')
        >>> y.dtype
        <type 'numpy.int32'>
        """
        res = zeros(shape=self.shape, ctx=self.context,
                    dtype=dtype, stype=self.stype)
        self.copyto(res)
        return res

    # TODO
    def copyto(self, other):
        """Copies the value of this array to another array.

        Parameters
        ----------
        other : NDArray or CSRNDArray or RowSparseNDArray or Context
            The destination array or context.

        Returns
        -------
        NDArray or CSRNDArray or RowSparseNDArray
            The copied array.
        """
        if isinstance(other, NDArray):
            if other.handle is self.handle:
                warnings.warn('You are attempting to copy an array to itself', RuntimeWarning)
                return
            return _internal._copyto(self, out=other)
        elif isinstance(other, Context):
            hret = _ndarray_cls(_new_alloc_handle(self.stype, self.shape, other,
                                                  True, self.dtype, self._aux_types))
            return _internal._copyto(self, out=hret)
        else:
            raise TypeError('copyto does not support type ' + str(type(other)))

