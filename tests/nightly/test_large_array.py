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

import mxnet as mx
import numpy as np
from mxnet import gluon, nd

# dimension constants
MEDIUM_X = 10000
LARGE_X = 100000000
LARGE_Y = 50000000
SMALL_Y = 50
LARGE_SIZE = LARGE_X * SMALL_Y

def test_gluon_embedding():
    m = gluon.nn.Embedding(SMALL_Y, MEDIUM_X)
    m.initialize()
    a = nd.zeros((MEDIUM_X, SMALL_Y))
    b = m(a)
    assert b.shape == (MEDIUM_X, SMALL_Y, MEDIUM_X)
    assert b.asnumpy().size == LARGE_SIZE

def test_ndarray_zeros():
    a = nd.zeros(shape=(LARGE_X, SMALL_Y))
    assert a[-1][0] == 0
    assert a.shape == (LARGE_X, SMALL_Y)
    assert a.size == LARGE_SIZE

def test_ndarray_ones():
    a = nd.ones(shape=(LARGE_X, SMALL_Y))
    assert a[-1][0] == 1
    assert nd.sum(a).asnumpy() == LARGE_SIZE

def test_ndarray_random_uniform():
    a = nd.random.uniform(shape=(LARGE_X, SMALL_Y))
    assert a[-1][0] != 0

def test_ndarray_empty():
    a = nd.empty((LARGE_X, SMALL_Y))
    assert a.shape == (LARGE_X, SMALL_Y)

def test_elementwise():
    a = nd.ones(shape=(LARGE_X, SMALL_Y))
    b = nd.ones(shape=(LARGE_X, SMALL_Y))
    res = a + b
    assert np.sum(res[-1].asnumpy() == 2) == a.shape[1]
    res = a + 1
    assert np.sum(res[-1].asnumpy() == 2) == a.shape[1]
    res = nd.sqrt(a + 3)
    assert np.sum(res[-1].asnumpy() == 2) == a.shape[1]

def test_reduce():
    a = nd.ones(shape=(LARGE_X, SMALL_Y)) 
    assert nd.sum(a).asnumpy() == a.shape[0] * a.shape[1]

def test_dot():
    a = nd.ones(shape=(LARGE_X, SMALL_Y)) 
    b = nd.ones(shape=(SMALL_Y, SMALL_Y))
    res = nd.dot(a, b)
    assert np.sum(res[-1].asnumpy() == SMALL_Y) == b.shape[1]

def test_FullyConnected():
    a = nd.ones(shape=(LARGE_X, SMALL_Y)) 
    b = nd.ones(shape=(SMALL_Y, SMALL_Y)) 
    res = nd.FullyConnected(a, b, num_hidden=b.shape[1], no_bias=True)
    assert np.sum(res[-1].asnumpy() == SMALL_Y) == b.shape[1]

def test_broadcast():
    a = nd.ones(shape=(LARGE_X, SMALL_Y))
    b = nd.arange(0, LARGE_X).reshape(LARGE_X, 1)
    res = nd.broadcast_to(b, shape=(b.shape[0], SMALL_Y))
    assert np.sum(res[-1].asnumpy() == LARGE_X) == res.shape[1]
    res = mx.nd.broadcast_like(b, a)
    assert np.sum(res[-1].asnumpy() == LARGE_X) == a.shape[1]

def test_clip():
    a = nd.arange(0, LARGE_X).reshape(LARGE_X, 1)
    b = nd.broadcast_to(a, shape=(a.shape[0], SMALL_Y))
    res = nd.clip(b, a_min=100, a_max=1000)
    assert np.sum(res[-1].asnumpy() == 1000) == b.shape[1]

def test_take():
    a = nd.ones(shape=(LARGE_X, SMALL_Y))
    idx = nd.arange(LARGE_X-1000, LARGE_X)
    res = nd.take(a, idx)
    assert np.sum(res[-1].asnumpy() == 1) == res.shape[1]

def test_slice():
    a = nd.ones(shape=(LARGE_X, SMALL_Y))
    res = nd.slice(a, begin=(LARGE_X-1000, 1), end=(LARGE_X, SMALL_Y))
    assert np.sum(res[-1].asnumpy() == 1) == res.shape[1]

def test_slice_assign():
    a = nd.ones(shape=(LARGE_X, SMALL_Y))
    a[LARGE_X-1:LARGE_X] = 1000
    assert np.sum(a[-1].asnumpy() == 1000) == a.shape[1]
 
def test_expand_dims():
    a = nd.ones(shape=(LARGE_X, SMALL_Y))
    res = nd.expand_dims(a, axis=1)
    assert res.shape == (a.shape[0], 1, a.shape[1])

def test_squeeze():
    a = nd.ones(shape=(LARGE_X, SMALL_Y))
    data = nd.expand_dims(a, axis=1)
    res = nd.squeeze(data)
    assert res.shape == a.shape

def test_broadcast_div():
    a = nd.ones(shape=(LARGE_X, SMALL_Y))
    b = nd.ones(shape=(LARGE_X, 1)) * 2
    res = a / b
    assert np.sum(res[-1].asnumpy() == 0.5) == a.shape[1]

def test_Dense(ctx=mx.cpu(0)):
    data = mx.nd.ones(shape=(50*1000*1000, 100))
    linear = gluon.nn.Dense(100)
    linear.initialize(ctx=ctx)
    res = linear(data)
    res.wait_to_read()
    assert res.shape == (50000000, 100)

def test_where():
    a = nd.ones(shape=(LARGE_X, SMALL_Y))
    b = nd.arange(0, LARGE_X).reshape(LARGE_X, 1)
    b = nd.broadcast_to(b, shape=(b.shape[0], SMALL_Y))
    res = nd.where(b > 100, a, b)
    assert np.sum(res[-1].asnumpy() == 1) == b.shape[1]

    csr_cond = nd.sparse.cast_storage(b < 10, 'csr')
    res = nd.sparse.where(csr_cond, a, b)
    assert np.sum(res[0].asnumpy() == 1) == b.shape[1]

def test_pick():
    a = mx.nd.ones(shape=(256*35, 1024*1024))
    b = mx.nd.ones(shape=(256*35,))
    res = mx.nd.pick(a,b)
    assert res.shape == b.shape

if __name__ == '__main__':
    import nose
    nose.runmodule()
