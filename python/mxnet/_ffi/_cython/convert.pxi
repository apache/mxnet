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

"""Acknowledgement: This file originates from incubator-tvm"""

from libc.stdint cimport *
from numbers import Integral

cdef extern from "mxnet/runtime/ffi_helper.h" namespace "mxnet::runtime":
    cdef cppclass Object:
        pass

    cdef cppclass ObjectPtr[T]:
        pass

    cdef cppclass ObjectRef:
        const Object* get() const

    cdef cppclass ADT(ObjectRef):
        ADT()

    cdef cppclass ADTBuilder:
        ADTBuilder()
        ADTBuilder(uint32_t tag, uint32_t size)
        void EmplaceInit(size_t idx, ObjectRef)
        ADT Get()

    cdef cppclass Integer(ObjectRef):
        Integer()
        Integer(int64_t)

    cdef cppclass Float(ObjectRef):
        Float()
        Float(double)
