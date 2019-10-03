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
import numpy as _np
from mxnet.test_utils import same, rand_shape_nd
from mxnet.runtime import Features
from common import with_seed

_features = Features()

@with_seed()
def test_tvm_broadcast_add():
    if _features.is_enabled("TVM_OP"):
        configs = [
            [[5, 6, 7, 8, 9], [1]],
            [[6, 4, 5, 2, 1], [6, 1, 5, 1, 1]],
            [[3, 5, 6], [1, 6]],
            [[3, 5, 6], [5, 1]],
            [[3, 5, 6], [5, 6]],
            [[4, 3, 2, 1], [2, 1]],
            [[4, 3, 2, 2], [4, 1, 1, 2]],
            [[6, 6], [6, 6]],
        ]
        for config in configs:
            a_shape = config[0]
            b_shape = config[1]
            a = mx.nd.normal(shape=a_shape)
            b = mx.nd.normal(shape=b_shape)
            a.attach_grad()
            b.attach_grad()
            with mx.autograd.record():
                c = mx.nd.contrib.tvm_vadd(a, b)
            c_np = a.asnumpy() + b.asnumpy()
            assert same(c.asnumpy(), c_np)
            # test backward
            c.backward()
            expected_grad_a = _np.ones_like(a.asnumpy()) * c_np.size / a.asnumpy().size
            expected_grad_b = _np.ones_like(b.asnumpy()) * c_np.size / b.asnumpy().size
            assert same(a.grad.asnumpy(), expected_grad_a)
            assert same(b.grad.asnumpy(), expected_grad_b)
            # test kAddTo request
            a = mx.nd.normal(shape=a_shape)
            b = mx.nd.normal(shape=b_shape)
            a.attach_grad()
            b.attach_grad()
            with mx.autograd.record():
                c = mx.nd.contrib.tvm_vadd(a, b)
                d = mx.nd.contrib.tvm_vadd(a, b)
            mx.autograd.backward([c, d])
            expected_grad_a = 2 * _np.ones_like(a.asnumpy()) * c.size / a.size
            expected_grad_b = 2 * _np.ones_like(b.asnumpy()) * c.size / b.size
            assert same(a.grad.asnumpy(), expected_grad_a)
            assert same(b.grad.asnumpy(), expected_grad_b)

if __name__ == '__main__':
    import nose
    nose.runmodule()
