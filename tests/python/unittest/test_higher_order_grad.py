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
from mxnet import nd, autograd
from mxnet.test_utils import assert_almost_equal, random_arrays
from common import with_seed


@with_seed()
def test_sin():
    def sin(x):
        return nd.sin(x)

    x = nd.array([1, 2, 3])
    expect_grad = -nd.sin(x)
    check_second_order_unary(x, sin, expect_grad)


@with_seed()
def test_cos():
    def cos(x):
        return nd.cos(x)

    x = nd.array([1, 2, 3])
    expect_grad = -nd.cos(x)
    check_second_order_unary(x, cos, expect_grad)


@with_seed()
def test_negative():
    def negative(x):
        return nd.negative(x)

    x = nd.array([1, 2, 3])
    expect_grad = nd.zeros_like(x)
    check_second_order_unary(x, negative, expect_grad)


@with_seed()
def test_relu():
    def relu(x):
        return nd.relu(x)

    x = nd.array([1, 2, 3])
    expect_grad = nd.zeros_like(x)
    check_second_order_unary(x, relu, expect_grad)




@with_seed()
def test_log():
    def log(x):
        return nd.log(x)

    def grad_grad_op(x):
        return -1/(x**2)

    arrays = random_arrays((2, 2), (2, 3), (4, 5, 2), (3, 1, 4, 5))

    for array in arrays:
        check_second_order_unary(array, log, grad_grad_op)


@with_seed()
def test_log2():
    def log2(x):
        return nd.log2(x)

    def grad_grad_op(x):
        return -1/((x**2) * math.log(2))

    arrays = random_arrays((2, 2), (2, 3), (4, 5, 2), (3, 1, 4, 5))

    for array in arrays:
        check_second_order_unary(array, log2, grad_grad_op)


@with_seed()
def test_log10():
    def log10(x):
        return nd.log10(x)

    def grad_grad_op(x):
        return -1/((x**2) * math.log(10))

    arrays = random_arrays((2, 2), (2, 3), (4, 5, 2), (3, 1, 4, 5))

    for array in arrays:
        check_second_order_unary(array, log10, grad_grad_op)


def check_second_order_unary(x, op, grad_grad_op):
    x = nd.array(x)
    expect_grad_grad = grad_grad_op(x)
    x.attach_grad()
    with autograd.record():
        y = op(x)
        y_grad = autograd.grad(y, x, create_graph=True, retain_graph=True)[0]
    y_grad.backward()
    assert_almost_equal(expect_grad_grad.asnumpy(), x.grad.asnumpy())


if __name__ == '__main__':
    import nose
    nose.runmodule()
