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
from ..ndarray_doc import _build_doc
from libc.stdint cimport uint32_t, int64_t

include "./base.pyi"

cdef class CythonTestClass:
    """Symbol is symbolic graph."""
    # handle for symbolic operator.
    cdef int cwritable

    def __init__(self):
        self.cwritable = 99

    def print_something(self, char *the_string):
      print('CythonTestClass::print_something( {} )'.format(the_string))

def print_pi(terms):
    print(float(0.0))

