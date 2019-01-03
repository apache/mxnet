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

from __future__ import print_function
import numpy as np
import mxnet as mx
import os


def binary_op_ex(sym, x_shape, y_shape):
    output_names = []
    def get_output_names_callback(name, arr):
        output_names.append(py_str(name))

    np.random.seed(0)
    x_npy = np.random.randint(0, 10, size=x_shape).astype(np.float32)
    y_npy = np.random.randint(0, 10, size=y_shape).astype(np.float32)
    exe = sym.simple_bind(ctx=mx.cpu(), x=x_shape, y=y_shape)
    exe.set_monitor_callback(get_output_names_callback)
    mx_out = exe.forward(is_train=True, x=x_npy, y=y_npy)[0].asnumpy()
    exe.backward()
    if ('MXNET_SUBGRAPH_BACKEND' in os.environ and 
        os.environ['MXNET_SUBGRAPH_BACKEND'] == "ngraph"):
        assert any(['ngraph' in name for name in output_names])
    return mx_out


def test_broadcast_op_no_head_grad():
    x = mx.symbol.Variable("x")
    y = mx.symbol.Variable("y")
    z = mx.sym.broadcast_not_equal(x, y)
    binary_op_ex(z, (1, 10), (10, 1))


def test_broadcast_mix_logic_op():
    x_shape = (1, 10)
    y_shape = (10, 1)
    x = mx.symbol.Variable("x")
    y = mx.symbol.Variable("y")
    z1 = mx.sym.broadcast_mul(x, y)
    z2 = mx.sym.broadcast_not_equal(z1, y)
    z3 = mx.sym.broadcast_mul(z1, z2)
    z4 = mx.sym.broadcast_equal(z1, z3)
    z5 = mx.sym.broadcast_not_equal(z3, z4)
    z6 = mx.sym.broadcast_mul(z5, z4)
    z = mx.sym.broadcast_equal(z6, x)

    binary_op_ex(z, (1, 10), (10, 1))

def test_batch_normalized_softmax_grad():
    xpu = mx.cpu()
    x = mx.sym.Variable('x')
    label = mx.sym.Variable('label')
    x_nd = mx.nd.array([[1, 6, 4, 2],[1, 6, 4, 2]], ctx=xpu)
    grad_x = mx.nd.zeros((2,4), ctx=xpu)
    label_nd = mx.nd.array([1,1], ctx=xpu)

    sym = mx.sym.SoftmaxOutput(data=x, label=label, ignore_label=0, 
                               use_ignore=False, normalization="batch")
    ex = sym.bind(ctx=xpu, args={'x': x_nd, 'label': label_nd}, 
                  args_grad={'x': grad_x})

    ex.forward(is_train=True)
    softmax_out = ex.outputs[0].asnumpy()
    expected_softmax_out = [[0.005806628, 0.861780069, 0.116629249, 0.015784052], 
                            [0.005806628, 0.861780069, 0.116629249, 0.015784052]]
    assert np.isclose(softmax_out, expected_softmax_out).all()

    ex.backward(is_train=True)
    grad_out = ex.grad_arrays[0].asnumpy()
    k = int(label_nd[0].asscalar())
    expected_grad_out = np.zeros((2,4))
    expected_grad_out[:, k] = - 1
    assert np.isclose(grad_out , (expected_softmax_out + expected_grad_out) / 2).all()

def test_valid_normalized_softmax_grad():
    xpu = mx.cpu()
    x = mx.sym.Variable('x')
    label = mx.sym.Variable('label')
    x_nd = mx.nd.array([[1, 6, 4, 2],[1, 6, 4, 2]], ctx=xpu)
    grad_x = mx.nd.zeros((2,4), ctx=xpu)
    label_nd = mx.nd.array([1,1], ctx=xpu)

    sym = mx.sym.SoftmaxOutput(data=x, label=label, ignore_label=0, 
                               use_ignore=True, normalization="valid")
    ex = sym.bind(ctx=xpu, args={'x': x_nd, 'label': label_nd}, 
                  args_grad={'x': grad_x})

    ex.forward(is_train=True)
    softmax_out = ex.outputs[0].asnumpy()
    expected_softmax_out = [[0.005806628, 0.861780069, 0.116629249, 0.015784052], 
                            [0.005806628, 0.861780069, 0.116629249, 0.015784052]]
    assert np.isclose(softmax_out, expected_softmax_out).all()

    ex.backward(is_train=True)
    grad_out = ex.grad_arrays[0].asnumpy()
    k = int(label_nd[0].asscalar())
    expected_grad_out = np.zeros((2,4))
    expected_grad_out[:, k] = - 1
    
    assert np.isclose(grad_out, (expected_softmax_out + expected_grad_out) 
                                 / sum(label_nd.asnumpy() != 0)).all()

def test_valid_make_loss():
    xpu = mx.cpu()
    x = mx.sym.Variable('x')
    label = mx.sym.Variable('label')
    x_nd = mx.nd.array([[0, 1, 1, 0], 
                        [1, 1, 1, .1]], ctx=xpu)
    grad_x = mx.nd.zeros((2,4), ctx=xpu)
    label_nd = mx.nd.array([1,1], ctx=xpu)

    sym = mx.sym.MakeLoss(x, normalization="valid", valid_thresh=0.2)
    ex = sym.bind(ctx=xpu, args={'x': x_nd, 'label': label_nd}, 
                  args_grad={'x': grad_x})

    ex.forward(is_train=True)
    out = ex.outputs[0].asnumpy()
    expected_out = [[0, 1, 1, 0], 
                    [1, 1, 1, .1]]
    assert np.isclose(out, expected_out).all()

    ex.backward(is_train=True)
    grad_out = ex.grad_arrays[0].asnumpy()
    expected_grad_out = np.ones((2,4))/5
    
    assert np.isclose(grad_out, expected_grad_out).all() 

def test_stop_gradient():                                    
    v1 = mx.nd.array([[1, 2]])                                 
    v2 = mx.nd.array([[0, 1]])                                 
    a = mx.sym.Variable('a')                                   
    b = mx.sym.Variable('b')                                   
    b_stop_grad = mx.sym.stop_gradient(3 * b)                  
    loss = mx.sym.MakeLoss(b_stop_grad + a)                    
                                                               
    executor = loss.simple_bind(ctx=mx.cpu(), a=(1,2), b=(1,2))
    executor.forward(is_train=True, a=v1, b=v2)                     
    assert np.isclose(executor.outputs[0].asnumpy(), [1,5]).all()
    executor.backward()                                  
    assert np.isclose(executor.grad_arrays[0].asnumpy(), [0,0]).all()
    assert np.isclose(executor.grad_arrays[1].asnumpy(), [1,1]).all()

if __name__ == '__main__':
    import nose
    nose.runmodule()