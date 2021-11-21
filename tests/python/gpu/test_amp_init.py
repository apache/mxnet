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

from contextlib import contextmanager
import ctypes

import numpy as np
import pytest

import mxnet as mx
from mxnet import amp
from mxnet.base import check_call, _LIB
from mxnet.gluon import nn
from mxnet.test_utils import assert_allclose


@pytest.fixture
def np_shape_array():
    flags = mx.npx.is_np_shape(), mx.npx.is_np_array(), mx.npx.is_np_default_dtype()
    mx.npx.set_np()
    yield
    mx.npx.set_np(*flags)


@pytest.fixture(scope='module')
def amp_init():
    amp.init()


@contextmanager
def optimize_layout(optimize=True):
    prev = ctypes.c_bool()
    check_call(_LIB.MXGetOptimizeLayout(ctypes.byref(prev)))
    check_call(_LIB.MXSetOptimizeLayout(ctypes.c_bool(optimize)))
    try:
        yield
    finally:
        check_call(_LIB.MXSetOptimizeLayout(prev))


def test_npi_concatenate_multicast(np_shape_array, amp_init):
    class Foo(nn.HybridBlock):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.dense0 = nn.Dense(16, in_units=8)

        def forward(self, x):
            y = self.dense0(x)
            return mx.np.concatenate([y, x], axis=-1)

    foo = Foo()
    foo.initialize(ctx=mx.gpu())

    data = mx.np.ones((32, 8), ctx=mx.gpu())
    out = foo(data)
    assert out.dtype == np.float32


class Conv(nn.HybridBlock):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.conv2d = nn.Conv2D(10, 3)

    def forward(self, x):
        y = self.conv2d(x)
        return y * 2


class ConvBN(nn.HybridBlock):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.conv2d = nn.Conv2D(10, 3)
        self.bn = nn.BatchNorm()

    def forward(self, x):
        y = self.conv2d(x)
        y = self.bn(y)
        return y * 2


class PoolConv(nn.HybridBlock):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pool = nn.MaxPool2D()
        self.conv2d = nn.Conv2D(10, 3)

    def forward(self, x):
        y = self.pool(x)
        y = self.conv2d(y)
        return y * 2


@pytest.mark.parametrize('model', [Conv, ConvBN, PoolConv])
def test_optimize_layout(np_shape_array, amp_init, model):
    m = model()
    m.initialize(ctx=mx.gpu())
    m.hybridize()
    x = mx.np.ones((32, 2, 20, 20), ctx=mx.gpu())
    y = m(x)
    params = {k:v.data() for k, v in m.collect_params().items()}
    with optimize_layout():
        m2 = model()
        m2.initialize(ctx=mx.gpu())
        m2.load_dict(params, device=mx.gpu())
        m2.hybridize()
        y2 = m2(x)
    assert_allclose(y2, y)
