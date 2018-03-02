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
from mxnet import gluon
from common import setup_module, with_seed
from mxnet.gluon import nn
from mxnet.base import MXNetError
from mxnet.test_utils import assert_exception, default_context, set_default_context
from nose.tools import assert_raises

@with_seed()
def test_exc_imperative():
    def imperative(exec_numpy=True):
        a = mx.nd.random.normal(0, 1, (2, 2))
        b = mx.nd.random.normal(0, -1, (2, 2))
        c = mx.nd.dot(a, b)
        if exec_numpy:
            c.asnumpy()

    imperative(exec_numpy=False)
    assert_raises(MXNetError, imperative, True)

@with_seed()
def test_exc_symbolic():
    def symbolic(exec_backward=True):
        x = mx.sym.Variable('x')
        y = mx.sym.Variable('y')
        z = mx.sym.Variable('z')
        x_shape = (2, 2)
        z_shape = (3, 2)
        inputs = [x, y]
        out = mx.symbol.ElementWiseSum(*inputs, name="esum")
        out = mx.sym.dot(z, out)
        out2 = mx.sym.random.normal(0, -1, x_shape, ctx=default_context())
        out = mx.sym.dot(out, out2)
        out = mx.sym.make_loss(out)
        arr = {'x': mx.nd.random.normal(0, 1, x_shape, ctx=default_context()),
               'y': mx.nd.random.normal(0, 1, x_shape, ctx=default_context()),
               'z': mx.nd.random.normal(0, 1, z_shape, ctx=default_context())}
        arr_grad = {'x': mx.nd.empty(x_shape), 'y': mx.nd.empty(x_shape), 'z': mx.nd.empty(z_shape)}
        exec1 = out.bind(ctx=default_context(), args=arr, args_grad=arr_grad)
        outputs = exec1.forward()
        if exec_backward:
            exec1.backward()
            exec1.grad_arrays[0].asnumpy()
        else:
            outputs[0].asnumpy()

    assert_raises(MXNetError, symbolic, False)
    assert_raises(MXNetError, symbolic, True)

@with_seed()
def test_exc_gluon():
    def gluon(exec_wait=True):
        model = nn.Sequential()
        model.add(nn.Dense(128, activation='tanh', in_units=10, flatten=False))
        model.add(nn.Dropout(1))
        model.add(nn.Dense(64, activation='tanh', in_units=256),
                  nn.Dense(32, in_units=64))
        x = mx.sym.var('data')
        y = model(x)
        model.collect_params().initialize(ctx=[default_context()])
        z = model(mx.nd.random.normal(10, -10, (32, 2, 10), ctx=default_context()))
        if exec_wait:
            z.wait_to_read()

    gluon(exec_wait=False)
    assert_raises(MXNetError, gluon, True)

@with_seed()
def test_exc_multiple_waits():
    caught = False
    try:
        a = mx.nd.random.normal(0, -1, (2, 2)).copyto(default_context())
        a.wait_to_read()
    except MXNetError:
        caught = True
    assert caught, "No exception thrown"
    try:
        b = mx.nd.random.normal(0, -1, (2, 2)).copyto(default_context())
        b.wait_to_read()
    except MXNetError:
        caught = True
    assert caught, "No exception thrown"

@with_seed()
def test_exc_post_fail():
    caught = False
    try:
        a, b = mx.nd.random_normal(0, -1, (2, 2)).copyto(default_context())
        a.asnumpy()
    except MXNetError:
        caught = True
    assert caught, "No exception thrown"
    b.asnumpy()

@with_seed()
def test_exc_mutable_var_fail():
    def mutable_var_check():
        a, b = mx.nd.random_normal(0, -1, (2, 2)).copyto(default_context())
        a = mx.nd.dot(a, a)
        a.asnumpy()
    assert_raises(MXNetError, mutable_var_check)

if __name__ == '__main__':
    import nose
    nose.runmodule()
