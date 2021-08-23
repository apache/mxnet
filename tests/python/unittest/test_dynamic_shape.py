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

import numpy as np
import mxnet as mx
import mxnet.ndarray.numpy._internal as _npi
from mxnet import gluon
from numpy.testing import assert_allclose, assert_array_equal
from mxnet.test_utils import *
from mxnet.base import _as_list
from mxnet.attribute import AttrScope


@mx.util.use_np
def test_dynamic_shape():

    class _TestBlock(gluon.HybridBlock):

        def __init__(self):
            super(_TestBlock, self).__init__()

        def forward(self, data, index):
            return _npi.boolean_mask(data, index)

    block = _TestBlock()
    block.hybridize()
    data = mx.np.array([[1, 2, 3],[4, 5, 6],[7, 8, 9]])
    index = mx.np.array([0, 1, 1])
    data.attach_grad()
    with mx.autograd.record():
        result = block(data, index)
    result.backward()
    result_nd = np.array([[4, 5, 6], [7, 8, 9]])
    data_grad_nd = np.array([[0., 0., 0.], [1., 1., 1.], [1., 1., 1.]])
    assert_almost_equal(result.asnumpy(), result_nd)
    assert_almost_equal(data.grad.asnumpy(), data_grad_nd)

@mx.util.use_np
def test_dynamic_shape_with_reshape():
    # test dynamic shape op followed by reshape op
    class _TestBlock(gluon.HybridBlock):

        def __init__(self):
            super(_TestBlock, self).__init__()

        def forward(self, data, index):
            return _npi.boolean_mask(data, index).reshape((-1, ))

    block = _TestBlock()
    block.hybridize()
    data = mx.np.array([[1, 2, 3],[4, 5, 6],[7, 8, 9]])
    index = mx.np.array([0, 1, 1])
    data.attach_grad()
    with mx.autograd.record():
        result = block(data, index)
    result.backward()
    result_nd = np.array([4, 5, 6, 7, 8, 9])
    data_grad_nd = np.array([[0., 0., 0.], [1., 1., 1.], [1., 1., 1.]])
    assert_almost_equal(result.asnumpy(), result_nd)
    assert_almost_equal(data.grad.asnumpy(), data_grad_nd)

@mx.util.use_np
def test_dynamic_shape_multiple_hybridize():
    # test multiple hybridize calls for the same block
    class _TestBlock(gluon.HybridBlock):

        def __init__(self):
            super(_TestBlock, self).__init__()

        def forward(self, data, index):
            return mx.np.sum(_npi.boolean_mask(data, index)) - 5

    block = _TestBlock()
    data = mx.np.array([[1, 2, 3],[4, 5, 6],[7, 8, 9]])
    index = mx.np.array([0, 1, 0])
    result_nd = np.array([10])

    block.hybridize()
    result = block(data, index)
    assert_almost_equal(result.asnumpy(), result_nd)

    block.hybridize(static_alloc=True)
    result = block(data, index)
    assert_almost_equal(result.asnumpy(), result_nd)

    block.hybridize(static_alloc=True, static_shape=True)
    result = block(data, index)
    assert_almost_equal(result.asnumpy(), result_nd)

@mx.util.use_np
def test_dynamic_shape_switch_hybridize():
    # test hybridize switch on and off for the same block 
    class _TestBlock(gluon.HybridBlock):
        def __init__(self):
            super(_TestBlock, self).__init__()

        def forward(self, data, index):
            return mx.np.sum(_npi.boolean_mask(data, index)) - 5

    block = _TestBlock()
    data = mx.np.array([[1, 2, 3],[4, 5, 6],[7, 8, 9]])
    index = mx.np.array([0, 1, 0])
    result_nd = np.array([10])

    block.hybridize()
    result = block(data, index)
    assert_almost_equal(result.asnumpy(), result_nd)

    block.hybridize(active=False)
    result = block(data, index)
    assert_almost_equal(result.asnumpy(), result_nd)

    block.hybridize(static_alloc=True, static_shape=True)
    result = block(data, index)
    assert_almost_equal(result.asnumpy(), result_nd)

@mx.util.use_np
def test_dynamic_shape_backward():
    # test dynamic shape ops with backward prop
    class _TestBlock(gluon.HybridBlock):
        def __init__(self):
            super(_TestBlock, self).__init__()

        def forward(self, data, index):
            return mx.np.sum(_npi.boolean_mask(data, index)) - 5

    block = _TestBlock()
    for static_alloc in [True, False]:
        block.hybridize(static_alloc=static_alloc)
        data = mx.np.array([[1, 2, 3],[4, 5, 6],[7, 8, 9]])
        index = mx.np.array([0, 1, 0])
        data.attach_grad()
        with mx.autograd.record():
            result = block(data, index)
        result.backward()
        result_nd = np.array([10.])
        data_grad_nd = np.array([[0., 0., 0.], [1., 1., 1.], [0., 0., 0.]])
        assert_almost_equal(result.asnumpy(), result_nd)
        assert_almost_equal(data.grad.asnumpy(), data_grad_nd)

