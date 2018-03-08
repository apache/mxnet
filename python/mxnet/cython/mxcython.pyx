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

from __future__ import absolute_import as _abs

import sys as _sys
import ctypes as _ctypes
import numpy as np
import time
from ...ndarray_doc import _build_doc
from libc.stdint cimport uint32_t, int64_t

include "./base.pyi"

# C API functions
cdef extern from "../../../src/cython/cpp_api.h":
    int CythonPrintFromCPP(const char *foo);
    int Printf(const char *fmt, ...);
    int TrivialCPPCall(int var);
    unsigned long long TimeInMilliseconds();

# C++ Rectangle class
cdef extern from "../../../src/cython/cpp_api.h" namespace "shapes":
    cdef cppclass Rectangle:
        Rectangle() except +
        Rectangle(int, int, int, int) except +
        int x0, y0, x1, y1
        int getArea()
        void getSize(int* width, int* height)
        void move(int, int)


# Cython class: CythonTestClass
cdef class CythonTestClass:
    """Symbol is symbolic graph."""
    # handle for symbolic operator.
    cdef int cwritable

    def __init__(self):
        self.cwritable = 99

    def print_something(self, str the_string):
        print('BEFORE CythonPrintFromCPP')
        CALL(CythonPrintFromCPP("This is from C++"))
        print('AFTER CythonPrintFromCPP')
        print('CythonTestClass::print_something( {} )'.format(the_string))

def test_cpp_class():
    cdef int recArea
    rec_ptr = new Rectangle(1, 2, 3, 4)
    try:
        recArea = rec_ptr.getArea()
        Printf("Printf() call: Area: %d\n", recArea)
    finally:
        del rec_ptr     # delete heap allocated object

def test_perf(int count, int make_c_call):
  cdef unsigned long long start = TimeInMilliseconds()
  cdef int foo = 0
  cdef int i = 0
  while i < count:
    foo += i
    if foo > count:
      foo = 0
    if make_c_call != 0:
      TrivialCPPCall(0)
    i += 1
  cdef unsigned long long stop = TimeInMilliseconds()
  Printf("CYTHON: %d items took %f seconds\n", count, float(stop - start)/1000)

def print_pi(terms):
    print(float(0.0))
