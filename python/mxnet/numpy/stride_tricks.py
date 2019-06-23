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

"""Util functions with broadcast."""

from ..ndarray.ndarray import _get_broadcast_shape
from . import _op as _mx_np_op


__all__ = ['broadcast_arrays']


def _broadcast_shape(*args):
    shape = ()
    for arr in args:
        shape = _get_broadcast_shape(shape, arr.shape)
    return shape


def broadcast_arrays(*args):
    """
    Broadcast any number of arrays against each other.

    Parameters
    ----------
    `*args` : a list of ndarrays
        The arrays to broadcast.

    Returns
    -------
    broadcasted : list of arrays
        These arrays are copies of the original arrays unless that all the input
        arrays have the same shape, the input list of arrays are returned
        instead of a list of copies.
    """
    shape = _broadcast_shape(*args)

    if all(array.shape == shape for array in args):
        # Common case where nothing needs to be broadcasted.
        return args

    return [_mx_np_op.broadcast_to(array, shape) for array in args]
