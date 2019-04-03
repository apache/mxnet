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
from mxnet import gluon, nd, autograd
from mxnet.test_utils import assert_almost_equal
from tests.python.unittest.common import with_seed


@with_seed()
def test_elemwise_mul():
    x = nd.array([1, 2, 3])
    y = nd.zeros(3)
    x.attach_grad()
    with autograd.record():
        y = nd.elemwise_mul(x, x) 
        y_grad = autograd.grad(y, x, create_graph=True, retain_graph=True)[0]
    y_grad.backward()
    expect_grad = nd.array([2, 2, 2])
    assert_almost_equal(expect_grad.asnumpy(), x.grad.asnumpy())


@with_seed()
def test_sin():
    x = nd.array([1, 2, 3])
    x.attach_grad()
    with autograd.record():
        y = nd.sin(x)
        y_grad = autograd.grad(y, x, create_graph=True, retain_graph=True)[0]
    y_grad.backward()
    expect_grad = -nd.sin(x)
    assert_almost_equal(expect_grad.asnumpy(), x.grad.asnumpy())


@with_seed()
def test_cos():
    x = nd.array([1, 2, 3])
    x.attach_grad()
    with autograd.record():
        y = nd.cos(x)
        y_grad = autograd.grad(y, x, create_graph=True, retain_graph=True)[0]
    y_grad.backward()
    expect_grad = -nd.cos(x)
    assert_almost_equal(expect_grad.asnumpy(), x.grad.asnumpy())


if __name__ == '__main__':
    import nose
    nose.runmodule()
