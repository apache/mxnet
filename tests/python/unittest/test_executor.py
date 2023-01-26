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
from mxnet.test_utils import assert_almost_equal, environment


def check_bind_with_uniform(uf, gf, dim, sf=None, lshape=None, rshape=None):
    """check function consistency with uniform random numbers"""
    shape = tuple(np.random.randint(1, int(1000**(1.0/dim)), size=dim))
    lhs = mx.symbol.Variable('lhs')
    rhs = mx.symbol.Variable('rhs')
    if sf is not None:
        ret = sf(lhs, rhs)
    else:
        ret = uf(lhs, rhs)

    assert ret.list_arguments() == ['lhs', 'rhs']
    lshape = shape if lshape is None else lshape
    rshape = shape if rshape is None else rshape

    lhs_arr = mx.nd.array(np.random.uniform(-1, 1, lshape))
    rhs_arr = mx.nd.array(np.random.uniform(-1, 1, rshape))
    lhs_grad = mx.nd.empty(lshape)
    rhs_grad = mx.nd.empty(rshape)
    executor = ret._bind(mx.Context('cpu'),
                        args=[lhs_arr, rhs_arr],
                        args_grad=[lhs_grad, rhs_grad])

    exec3 = ret._bind(mx.Context('cpu'),
                     args=[lhs_arr, rhs_arr])


    exec4 = ret._bind(mx.Context('cpu'),
                     args={'rhs': rhs_arr, 'lhs': lhs_arr},
                     args_grad={'lhs': lhs_grad, 'rhs': rhs_grad})

    executor.forward()
    exec3.forward()
    exec4.forward()
    out2 = executor.outputs[0].asnumpy()
    out1 = uf(lhs_arr.asnumpy(), rhs_arr.asnumpy())
    out3 = exec3.outputs[0].asnumpy()
    out4 = exec4.outputs[0].asnumpy()
    assert_almost_equal(out1, out2, rtol=1e-5, atol=1e-5)
    assert_almost_equal(out1, out3, rtol=1e-5, atol=1e-5)
    assert_almost_equal(out1, out4, rtol=1e-5, atol=1e-5)
    # test gradient
    out_grad = mx.nd.array(np.ones(out2.shape))
    lhs_grad2, rhs_grad2 = gf(out_grad.asnumpy(),
                              lhs_arr.asnumpy(),
                              rhs_arr.asnumpy())
    executor.backward([out_grad])

    assert_almost_equal(lhs_grad.asnumpy(), lhs_grad2, rtol=1e-5, atol=1e-5)
    assert_almost_equal(rhs_grad.asnumpy(), rhs_grad2, rtol=1e-5, atol=1e-5)


def test_bind():
    for enable_bulking in ['0', '1']:
        with environment({'MXNET_EXEC_BULK_EXEC_INFERENCE': enable_bulking,
                          'MXNET_EXEC_BULK_EXEC_TRAIN': enable_bulking}):
            nrepeat = 10
            maxdim = 4
            for _ in range(nrepeat):
                for dim in range(1, maxdim):
                    check_bind_with_uniform(lambda x, y: x + y,
                                            lambda g, x, y: (g, g),
                                            dim)
                    check_bind_with_uniform(lambda x, y: x - y,
                                            lambda g, x, y: (g, -g),
                                            dim)
                    check_bind_with_uniform(lambda x, y: x * y,
                                            lambda g, x, y: (y * g, x * g),
                                            dim)
                    check_bind_with_uniform(lambda x, y: x / y,
                                            lambda g, x, y: (g / y, -x * g/ (y**2)),
                                            dim)

                    check_bind_with_uniform(lambda x, y: np.maximum(x, y),
                                            lambda g, x, y: (g * (x>=y), g * (y>x)),
                                            dim,
                                            sf=mx.symbol.maximum)
                    check_bind_with_uniform(lambda x, y: np.minimum(x, y),
                                            lambda g, x, y: (g * (x<=y), g * (y<x)),
                                            dim,
                                            sf=mx.symbol.minimum)


# @roywei: Removing fixed seed as flakiness in this test is fixed
# tracked at https://github.com/apache/mxnet/issues/11686
def test_dot():
    nrepeat = 10
    maxdim = 4
    for _ in range(nrepeat):
        s =tuple(np.random.randint(1, 200, size=3))
        check_bind_with_uniform(lambda x, y: np.dot(x, y),
                                lambda g, x, y: (np.dot(g, y.T), np.dot(x.T, g)),
                                2,
                                lshape=(s[0], s[1]),
                                rshape=(s[1], s[2]),
                                sf = mx.symbol.dot)
    for _ in range(nrepeat):
        s =tuple(np.random.randint(1, 200, size=1))
        check_bind_with_uniform(lambda x, y: np.dot(x, y),
                                lambda g, x, y: (g * y, g * x),
                                2,
                                lshape=(s[0],),
                                rshape=(s[0],),
                                sf = mx.symbol.dot)


def test_reshape():
    x = mx.sym.Variable('x')
    y = mx.sym.FullyConnected(x, num_hidden=4)

    exe = y._simple_bind(mx.cpu(), x=(5,4), grad_req='null')
    exe.arg_arrays[0][:] = 1
    exe.arg_arrays[1][:] = mx.nd.ones((4,4))
    exe.arg_arrays[2][:] = 0

    exe.forward(is_train=False)
    # test sub exec forward
    assert np.all(exe.outputs[0].asnumpy() == 4)
    # test shared memory
    assert np.all(exe.outputs[0].asnumpy()[:3] == 4)
    # test base exec forward
    exe.forward(is_train=False)
    assert np.all(exe.outputs[0].asnumpy() == 4)

    # data ndarray is not shared between exe and new_exe
    exe.arg_arrays[0][:] = 0
    # weight ndarray is shared between exe and new_exe
    assert np.all(exe.arg_arrays[1].asnumpy() == 1)

def test_cached_op_init():
    def check_init(static_alloc, static_shape):
        out = mx.sym.zeros((3,3))
        flags = [('static_alloc', static_alloc), ('static_shape', static_shape)]
        exe = mx.ndarray.CachedOp(out, flags)
        z = exe(None, default_device=mx.cpu())
        assert np.all(z.asnumpy() == 0)

    check_init(False, False)
    check_init(True, False)
    check_init(True, True)

def test_elemwise_add_grad():
    json = "{\"nodes\": [{\"op\":\"null\",\"name\":\".Inputs.Input1\",\"inputs\":[]},{\"op\":\"null\",\"name\":\".Inputs.Input2\",\"inputs\":[]},{\"op\":\"elemwise_add\",\"name\":\".$0\",\"inputs\":[[0,0,0],[1,0,0]]},{\"op\":\"_copy\",\"name\":\".Outputs.Output\",\"inputs\":[[2,0,0]]}],\"arg_nodes\":[0,1],\"heads\":[[3,0,0]]}"
    sym = mx.symbol.fromjson(json)

    ex = sym._bind(
        mx.cpu(), 
        {'.Inputs.Input1': mx.nd.array([0.4]), '.Inputs.Input2': mx.nd.array([0.5])},
        args_grad={
            '.Inputs.Input1': mx.ndarray.zeros((1)), 
            '.Inputs.Input2': mx.ndarray.zeros((1))
        },
        grad_req={'.Inputs.Input1': 'null', '.Inputs.Input2': 'write'}
    )
    ex.forward(is_train=True)
    print(ex.outputs)
    ex.backward(out_grads=mx.nd.array([1]))
    print(ex.grad_arrays)
