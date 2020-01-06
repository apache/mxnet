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

""" Adapted from incubator-tvm/python/tvm/_ffi/_cython/ndarray.pxi """

import ctypes
from ...numpy import ndarray

# cdef NewArray(NDArrayHandle handle, int stype=-1, int is_np_array=0):
#     """Create a new array given handle"""
#     create_array_fn = _np_ndarray_cls if is_np_array else _ndarray_cls
#     return create_array_fn(_ctypes.cast(<unsigned long long>handle, _ctypes.c_void_p), stype=stype)


cdef c_make_array(void* handle):
    # create_array_fn = _np_ndarray_cls
    # print(create_array_fn)
    # return return ndarray(handle=None if value[0].v_handle == 0 else ctypes.cast(value[0].v_handle, NDArrayHandle))
    # return ctypes.cast(<unsigned long long>handle, ctypes.c_void_p)
    return ndarray(handle=<unsigned long long>handle)
