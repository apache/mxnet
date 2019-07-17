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


import math
import random
from mxnet import nd, autograd
from mxnet.test_utils import assert_almost_equal, random_arrays, rand_shape_nd, same
from common import with_seed
import mxnet.autograd as ag
import mxnet.ndarray as nd
from mxnet import gluon
import mxnet
from nose.tools import ok_
import numpy as np


@with_seed()
def test_sin():
    def sin(x):
        return nd.sin(x)

    def grad_grad_op(x):
        return -nd.sin(x)

    for dim in range(1, 5):
        shape = rand_shape_nd(dim)
        array = random_arrays(shape)
        check_second_order_unary(array, sin, grad_grad_op)


@with_seed()
def test_cos():
    def cos(x):
        return nd.cos(x)

    def grad_grad_op(x):
        return -nd.cos(x)

    for dim in range(1, 5):
        shape = rand_shape_nd(dim)
        array = random_arrays(shape)
        check_second_order_unary(array, cos, grad_grad_op)


@with_seed()
def test_tan():
    def tan(x):
        return nd.tan(x)

    def grad_op(x):
        return 1 / nd.cos(x)**2

    def grad_grad_op(x):
        return 2 * tan(x) * grad_op(x)

    for dim in range(1, 5):
        shape = rand_shape_nd(dim)
        array = random_arrays(shape)
        check_second_order_unary(array, tan, grad_grad_op)


@with_seed()
def test_tanh():
    def tanh(x):
        return nd.tanh(x)

    def grad_op(x):
        return 1 / nd.cosh(x)**2

    def grad_grad_op(x):
        return -2 * tanh(x) * grad_op(x)

    for dim in range(1, 5):
        shape = rand_shape_nd(dim)
        array = random_arrays(shape)
        check_second_order_unary(
            array, tanh, grad_grad_op, rtol=1e-6, atol=1e-6)


@with_seed()
def test_arctan():
    def arctan(x):
        return nd.arctan(x)

    def grad_grad_op(x):
        return (-2 * x)/((1 + x**2)**2)

    for dim in range(1, 5):
        shape = rand_shape_nd(dim)
        array = random_arrays(shape)
        # Domain of arctan is all real numbers.
        # Scale std_dev
        array *= random.randint(500, 10000)
        check_second_order_unary(array, arctan, grad_grad_op)


@with_seed()
def test_arctanh():
    def arctanh(x):
        return nd.arctanh(x)

    def grad_grad_op(x):
        return (2 * x)/((1 - x**2)**2)

    for dim in range(1, 5):
        shape = rand_shape_nd(dim)
        array = random_arrays(shape)
        check_second_order_unary(array, arctanh, grad_grad_op)


@with_seed()
def test_radians():
    def radians(x):
        return nd.radians(x)

    def grad_grad_op(x):
        return nd.zeros_like(x)

    for dim in range(1, 5):
        shape = rand_shape_nd(dim)
        array = random_arrays(shape)
        check_second_order_unary(array, radians, grad_grad_op)


@with_seed()
def test_relu():
    def relu(x):
        return nd.relu(x)

    def grad_grad_op(x):
        return nd.zeros_like(x)

    for dim in range(1, 5):
        shape = rand_shape_nd(dim)
        array = random_arrays(shape)
        check_second_order_unary(array, relu, grad_grad_op)


@with_seed()
def test_log():
    def log(x):
        return nd.log(x)

    def grad_grad_op(x):
        return -1/(x**2)

    for dim in range(1, 5):
        shape = rand_shape_nd(dim)
        array = random_arrays(shape)
        check_second_order_unary(array, log, grad_grad_op)


@with_seed()
def test_log2():
    def log2(x):
        return nd.log2(x)

    def grad_grad_op(x):
        return -1/((x**2) * math.log(2))

    for dim in range(1, 5):
        shape = rand_shape_nd(dim)
        array = random_arrays(shape)
        check_second_order_unary(array, log2, grad_grad_op)


@with_seed()
def test_log10():
    def log10(x):
        return nd.log10(x)

    def grad_grad_op(x):
        return -1/((x**2) * math.log(10))

    for dim in range(1, 5):
        shape = rand_shape_nd(dim)
        array = random_arrays(shape)
        check_second_order_unary(array, log10, grad_grad_op)


@with_seed()
def test_reciprocal():
    def reciprocal(x):
        return nd.reciprocal(x)

    def grad_grad_op(x):
        return 2 / x**3

    for dim in range(1, 5):
        shape = rand_shape_nd(dim)
        array = random_arrays(shape)
        check_second_order_unary(array, reciprocal, grad_grad_op)


@with_seed()
def test_abs():
    def abs(x):
        return nd.abs(x)

    def grad_grad_op(x):
        return nd.zeros_like(x)

    for dim in range(1, 5):
        shape = rand_shape_nd(dim)
        array = random_arrays(shape)
        check_second_order_unary(array, abs, grad_grad_op)


@with_seed()
def test_clip():
    def clip(x):
        a_min, a_max = sorted([random.random(), random.random()])

        return nd.clip(x, a_min, a_max)

    def grad_grad_op(x):
        return nd.zeros_like(x)

    for dim in range(1, 5):
        shape = rand_shape_nd(dim)
        array = random_arrays(shape)
        check_second_order_unary(array, clip, grad_grad_op)


@with_seed()
def test_dropout():
    def dropout(x):
        return nd.Dropout(x)

    def grad_grad_op(x):
        return nd.zeros_like(x)

    for dim in range(1, 5):
        shape = rand_shape_nd(dim)
        array = random_arrays(shape)
        check_second_order_unary(array, dropout, grad_grad_op)


def test_sigmoid():
    def sigmoid(x):
        return nd.sigmoid(x)

    def grad_op(x):
        return sigmoid(x) * (1 - sigmoid(x))

    def grad_grad_op(x):
        return grad_op(x) * (1 - 2 * sigmoid(x))

    for dim in range(1, 5):
        shape = rand_shape_nd(dim)
        array = random_arrays(shape)
        check_second_order_unary(array, sigmoid, grad_grad_op)


@with_seed()
def test_sqrt():
    def sqrt(x):
        return nd.sqrt(x)

    def grad_grad_op(x):
        return -1/(4 * sqrt(x**3))

    sigma = random.randint(25, 100)
    mu = random.randint(500, 1000)

    for dim in range(1, 5):
        shape = rand_shape_nd(dim)
        array = random_arrays(shape)
        array = sigma * array + mu
        # Only positive numbers
        assert((array > 0).all())
        check_second_order_unary(array, sqrt, grad_grad_op)


@with_seed()
def test_cbrt():
    def cbrt(x):
        return nd.cbrt(x)

    def grad_grad_op(x):
        return -2/(9 * cbrt(x**5))

    sigma = random.randint(25, 100)
    mu = random.randint(500, 1000)

    for dim in range(1, 5):
        shape = rand_shape_nd(dim)
        array = random_arrays(shape)
        array = sigma * array + mu
        # Only positive numbers
        assert((array > 0).all())
        check_second_order_unary(array, cbrt, grad_grad_op)


def check_second_order_unary(x, op, grad_grad_op, rtol=None, atol=None):
    x = nd.array(x)
    grad_grad_x = grad_grad_op(x)
    x.attach_grad()

    # Manual head_grads.
    y_grad = nd.random.normal(shape=x.shape)
    head_grad_grads = nd.random.normal(shape=x.shape)

    # Perform compute.
    with autograd.record():
        y = op(x)
        x_grad = autograd.grad(heads=y, variables=x, head_grads=y_grad,
                               create_graph=True, retain_graph=True)[0]
    x_grad.backward(head_grad_grads)

    # Compute expected values.
    expected_grad_grad = grad_grad_x.asnumpy() * head_grad_grads.asnumpy() * \
        y_grad.asnumpy()

    # Validate the gradients.
    assert_almost_equal(expected_grad_grad,
                        x.grad.asnumpy(), rtol=rtol, atol=atol)

class RandomShapes(object):
    def __init__(self, dim, startdim=1):
        self.dim = dim
        self.curdim = startdim

    def __iter__(self):
        return self

    def next(self):
        return self.__next__()

    def __next__(self):
        if self.curdim > self.dim:
            raise StopIteration
        shape = rand_shape_nd(self.curdim)
        x = nd.random.normal(shape=shape)
        self.curdim += 1
        return x


@with_seed()
def test_dense_backward():
    for x in RandomShapes(4,2):
        print(x)
        net = gluon.nn.Sequential()
        with net.name_scope():
            net.add(gluon.nn.Dense(1))

        net.initialize(mxnet.initializer.Constant(.5))
        x.attach_grad()
        with ag.record():
            y = net.forward(x)
            params = [p.data() for p in net.collect_params().values()]
            print(params)
            x_grad = ag.grad(heads=y, variables=x, create_graph=True, retain_graph=True)[0]
        x_grad.backward()
        same(x.grad, nd.zeros(4))


def test_fc():
    x = nd.random.uniform(shape=(5,3,2))
    w = nd.random.uniform(shape=(8,6))
    b = nd.random.uniform(shape=(8,))
    x.attach_grad()
    w.attach_grad()
    ag.set_recording(True)
    y = nd.FullyConnected(data=x, weight=w, bias=b, flatten=True, num_hidden=8)
    #x_grad = ag.grad(y, x, create_graph=True, retain_graph=True)
    #x_grad_grad = ag.grad(x_grad, x, create_graph=False, retain_graph=True)
    w_grad = ag.grad(y, w, create_graph=True, retain_graph=True)[0]
    #w_grad.backward()
    #w_grad_grad=w_grad.grad
    w_grad_grad = ag.grad(w_grad, w, create_graph=False, retain_graph=True)[0]
    ag.set_recording(False)
    #print(y)
    #print(x_grad)
    print(w)
    #print(x_grad_grad)
    print("w_grad: {}".format(w_grad.shape))
    print("w_grad_grad: {}".format(w_grad_grad.shape))
    print(w_grad_grad)



if __name__ == '__main__':
    import nose
    nose.runmodule()
