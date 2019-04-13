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
from mxnet import gluon
from numpy.testing import assert_allclose, assert_array_equal
from mxnet.test_utils import *
from mxnet.base import _as_list
from mxnet.attribute import AttrScope
from common import with_seed


def test_dynamic_shape():

    class _TestBlock(gluon.HybridBlock):

        def __init__(self):
            super(_TestBlock, self).__init__()

        def hybrid_forward(self, F, data, index):
            return F.contrib.boolean_mask(data, index)

    block = _TestBlock()
    block.hybridize()
    data = mx.nd.array([[1, 2, 3],[4, 5, 6],[7, 8, 9]])
    index = mx.nd.array([0, 1, 1])
    data.attach_grad()
    with mx.autograd.record():
        result = block(data, index)
    result.backward()
    result_nd = np.array([[4, 5, 6], [7, 8, 9]])
    data_grad_nd = np.array([[0., 0., 0.], [1., 1., 1.], [1., 1., 1.]])
    assert_almost_equal(result.asnumpy(), result_nd)
    assert_almost_equal(data.grad.asnumpy(), data_grad_nd)


if __name__ == '__main__':
    import nose
    nose.runmodule()
