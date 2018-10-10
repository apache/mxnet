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

# pylint: skip-file
import mxnet as mx
import numpy as np
from common import models, with_seed
from mxnet import autograd
from nose.tools import *
from mxnet.test_utils import assert_almost_equal

@with_seed()
def test_infer_multiout_op():
    data = mx.nd.arange(16, dtype=np.float64).reshape((4, 4))
    data.attach_grad()

    with autograd.record():
        y = mx.nd.split(data, axis=0, num_outputs=2)
    y[0].backward()
    assert data.grad.dtype == np.float64

@with_seed()
def test_infer_multiout_op2():
    def test_func(a):
        q, l = mx.nd.linalg.gelqf(a)
        return mx.nd.sum(l)

    data32 = mx.nd.random.normal(shape=(2, 3), ctx=mx.cpu(), dtype=np.float32)
    data32.attach_grad()
    with autograd.record():
        test32 = test_func(data32)
        test32.backward()

    data64 = mx.nd.Cast(data32, dtype=np.float64)
    data64.attach_grad()
    with autograd.record():
        test64 = test_func(data64)
        test64.backward()
    assert_almost_equal(data64.grad.asnumpy(), data32.grad.asnumpy(), atol=1e-5, rtol=1e-5)


if __name__ == '__main__':
    import nose
    nose.runmodule()