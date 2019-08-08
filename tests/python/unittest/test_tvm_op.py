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
from mxnet.test_utils import same, rand_shape_nd
from mxnet.runtime import Features
from common import with_seed

_features = Features()

@with_seed()
def test_tvm_broadcast_add():
    if _features.is_enabled("TVM_OP"):
        a_shape = rand_shape_nd(4)
        b_shape = (1,) + a_shape[1:2] + (1, 1)
        a = mx.nd.normal(shape=a_shape)
        b = mx.nd.normal(shape=b_shape)
        c = mx.nd.contrib.tvm_vadd(a, b)
        c_np = a.asnumpy() + b.asnumpy()
        assert same(c.asnumpy(), c_np)

@with_seed()
def test_tvm_broadcast_fmax():
    if _features.is_enabled("TVM_OP"):
        a_shape = rand_shape_nd(4)
        b_shape = (1,) + a_shape[1:2] + (1, 1)
        a = mx.nd.normal(shape=a_shape)
        b = mx.nd.normal(shape=b_shape)
        c = mx.nd.contrib.tvm_fmax(a, b)
        c_np = np.fmax(a.asnumpy(), b.asnumpy())
        assert same(c.asnumpy(), c_np)

@with_seed()
def test_tvm_broadcast_fmin():
    if _features.is_enabled("TVM_OP"):
        a_shape = rand_shape_nd(4)
        b_shape = (1,) + a_shape[1:2] + (1, 1)
        a = mx.nd.normal(shape=a_shape)
        b = mx.nd.normal(shape=b_shape)
        c = mx.nd.contrib.tvm_fmin(a, b)
        c_np = np.fmin(a.asnumpy(), b.asnumpy())
        assert same(c.asnumpy(), c_np)

if __name__ == '__main__':
    import nose
    nose.runmodule()
