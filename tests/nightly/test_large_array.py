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
from mxnet import gluon, nd


class TestLargeArray(unittest.TestCase):
    def test_ndarray2numpy(self):
        m = gluon.nn.Embedding(14000, 128)
        m.initialize()
        ind = nd.zeros((700000, 128))
        x = m(ind)
        x.shape
        test = x.asnumpy()
        assert (x.shape == test.shape)
    
    def test_ndarray_ones(self):
        arr = nd.ones(shape=(100000000, 50))
        assert arr[-1][0] == 1
        assert nd.sum(arr).asnumpy() == 5000000000

    def test_ndarray_zeros(self):
        arr = nd.zeros(shape=(5000000000))
        assert arr.shape == (5000000000,)
        assert arr.size == 5000000000

    def test_ndarray_arrange(self):
        arr = mx.nd.arange(0, 5000000000, dtype='int64')
        assert arr[-1] == 4999999999
        assert mx.nd.slice(arr, begin=-2, end=-1) == 4999999998

    def test_ndarray_random_uniform(self):
        arr = mx.nd.random.uniform(shape=(100000000, 50))
        assert arr[-1][0] != 0

if __name__ == '__main__':
    unittest.main()
