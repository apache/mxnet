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

""" Adapted from incubator-tvm/python/tvm/_ffi/_cython/convert.pxi """

from libc.stdint cimport *
from numbers import Integral

cdef extern from "mxnet/runtime/object.h" namespace "mxnet::runtime":
    cdef cppclass Object:
        pass

    cdef cppclass ObjectPtr[T]:
        ObjectPtr()

    cdef cppclass ObjectRef:
        ObjectRef()
        ObjectRef(ObjectPtr[Object])
        Object* get()

    cdef ObjectPtr[T] GetObjectPtr[T](T* ptr)


cdef extern from "mxnet/runtime/container.h" namespace "mxnet::runtime":
    cdef cppclass ObjectRef:
        const Object* get() const
    cdef cppclass ADT(ObjectRef):
        ADT()


cdef extern from "mxnet/runtime/ffi_helper.h" namespace "mxnet::runtime":
    cdef cppclass ADTBuilder:
        ADTBuilder()
        ADTBuilder(uint32_t tag, uint32_t size)
        void EmplaceInit(size_t idx, ObjectRef)
        ADT Get()

    cdef ObjectRef CreateEllipsis()

    cdef cppclass Slice(ObjectRef):
        Slice()
        Slice(int)
        Slice(int, int, int)
    
    cdef cppclass Integer(ObjectRef):
        Integer()
        Integer(int)

    cdef int64_t SliceNoneValue()


cdef extern from "mxnet/runtime/memory.h" namespace "mxnet::runtime":
    cdef ObjectPtr[T] make_object[T]()


cdef inline ADT convert_tuple(tuple src_tuple) except *:
    cdef uint32_t size = len(src_tuple)
    cdef ADTBuilder builder = ADTBuilder(0, size)

    for i in range(size):
        builder.EmplaceInit(i, convert_object(src_tuple[i]))

    return builder.Get()


cdef inline ADT convert_list(list src) except *:
    cdef uint32_t size = len(src)
    cdef ADTBuilder builder = ADTBuilder(0, size)

    for i in range(size):
        builder.EmplaceInit(i, convert_object(src[i]))

    return builder.Get()

# cdef inline Slice convert_slice(slice slice_obj) except *:
#     cdef int64_t kNoneValue = SliceNoneValue()
#     return Slice(<int>(slice_obj.start) if slice_obj.start is not None else kNoneValue,
#                  <int>(slice_obj.stop) if slice_obj.stop is not None else kNoneValue,
#                  <int>(slice_obj.step) if slice_obj.step is not None else kNoneValue)


cdef inline ObjectRef convert_object(object src_obj) except *:
    if isinstance(src_obj, Integral):
        return Integer(<int>src_obj)
    elif isinstance(src_obj, tuple):
        return convert_tuple(src_obj)
    elif isinstance(src_obj, list):
        return convert_list(src_obj)
    # elif src_obj is Ellipsis:
    #     return CreateEllipsis()
    # elif isinstance(src_obj, slice):
    #     return convert_slice(src_obj)
    # elif src_obj is None:
    #     return ObjectRef()
    else:
        raise TypeError("Don't know how to convert type %s" % type(src_obj))
