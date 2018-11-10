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

import unittest
import mxnet as mx
import numpy as np
from mxnet import gluon, nd

# dimension constants
MEDIUM_X = 10000
LARGE_X = MEDIUM_X * MEDIUM_X
SMALL_Y = 50
LARGE_SIZE = LARGE_X * SMALL_Y

class TestLargeArray(unittest.TestCase):
    def test_gluon_embedding(self):
        m = gluon.nn.Embedding(SMALL_Y, MEDIUM_X)
        m.initialize()
        a = nd.zeros((MEDIUM_X, SMALL_Y))
        b = m(a)
        assert b.shape == (MEDIUM_X, SMALL_Y, MEDIUM_X)
        assert b.asnumpy().size == LARGE_SIZE

    def test_ndarray_zeros(self):
        a = nd.zeros(shape=(LARGE_X, SMALL_Y))
        assert a[-1][0] == 0
        assert a.shape == (LARGE_X, SMALL_Y)
        assert a.size == LARGE_SIZE

    def test_ndarray_ones(self):
        a = nd.ones(shape=(LARGE_X, SMALL_Y))
        assert a[-1][0] == 1
        assert nd.sum(a).asnumpy() == LARGE_SIZE

    def test_ndarray_zeros2(self):
        a = nd.zeros(shape=(LARGE_SIZE))
        assert a[LARGE_SIZE-1] == 0
        assert a.shape == (LARGE_SIZE,)

    def test_ndarray_arange(self):
        a = nd.arange(0, LARGE_SIZE, dtype='int64')
        assert a[-1] == LARGE_SIZE - 1
        assert nd.slice(a, begin=-2, end=-1) == (LARGE_SIZE - 2)

    def test_ndarray_random_uniform(self):
        a = nd.random.uniform(shape=(LARGE_X, SMALL_Y))
        assert a[-1][0] != 0

    def test_ndarray_empty(self):
        a = np.empty((LARGE_SIZE,))
        b = nd.array(a)
        assert b.shape == (LARGE_SIZE,)

if __name__ == '__main__':
    unittest.main()
