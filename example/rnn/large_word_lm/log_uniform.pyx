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

from libcpp.unordered_set cimport unordered_set
import cython

cdef extern from "LogUniformGenerator.h":
    cdef cppclass LogUniformGenerator:
        LogUniformGenerator(int) except +
        unordered_set[long] draw(int, int*) except +

cdef class LogUniformSampler:
    cdef LogUniformGenerator* c_sampler

    def __cinit__(self, N):
        self.c_sampler = new LogUniformGenerator(N)

    def __dealloc__(self):
        del self.c_sampler

    def sample_unique(self, size):
        cdef int num_tries
        samples = list(self.c_sampler.draw(size, &num_tries))
        return samples, num_tries
