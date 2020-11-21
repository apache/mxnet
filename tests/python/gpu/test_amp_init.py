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
from mxnet.gluon import nn
from mxnet import amp
import numpy as np
import pytest


@pytest.fixture
def np_shape_array():
    flags = mx.npx.is_np_shape(), mx.npx.is_np_array(), mx.npx.is_np_default_dtype()
    mx.npx.set_np()
    yield
    mx.npx.set_np(*flags)


@pytest.fixture(scope='module')
def amp_init():
    amp.init()


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
