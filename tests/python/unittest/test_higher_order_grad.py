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
from mxnet.test_utils import assert_almost_equal, random_arrays, rand_shape_nd
from common import with_seed


@with_seed()
def test_sin():
    def sin(x):
        return nd.sin(x)

    def grad_grad_op(x):
        return -nd.sin(x)

    def grad_grad_grad_op(x):
        return -nd.cos(x)

    for dim in range(1, 5):
        shape = rand_shape_nd(dim)
        array = random_arrays(shape)
        check_second_order_unary(array, sin, grad_grad_op)
        # TODO(kshitij12345): Remove
        check_nth_order_unary(array, sin,
                              [grad_grad_op, grad_grad_grad_op], [2, 3])


@with_seed()
def test_cos():
    def cos(x):
        return nd.cos(x)

    def grad_grad_op(x):
        return -nd.cos(x)

    def grad_grad_grad_op(x):
        return nd.sin(x)

    for dim in range(1, 5):
        shape = rand_shape_nd(dim)
        array = random_arrays(shape)
        check_second_order_unary(array, cos, grad_grad_op)
        # TODO(kshitij12345): Remove
        check_nth_order_unary(array, cos,
                              [grad_grad_op, grad_grad_grad_op], [2, 3])


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
def test_sinh():
    def sinh(x):
        return nd.sinh(x)

    def grad_grad_op(x):
        return sinh(x)

    for dim in range(1, 5):
        shape = rand_shape_nd(dim)
        array = random_arrays(shape)
        check_second_order_unary(array, sinh, grad_grad_op)


@with_seed()
def test_cosh():
    def cosh(x):
        return nd.cosh(x)

    def grad_grad_op(x):
        return cosh(x)

    for dim in range(1, 5):
        shape = rand_shape_nd(dim)
        array = random_arrays(shape)
        check_second_order_unary(array, cosh, grad_grad_op)


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

    def grad_op(x):
        return 1/x

    def grad_grad_op(x):
        return -1/(x**2)

    for dim in range(1, 5):
        shape = rand_shape_nd(dim)
        array = random_arrays(shape)
        check_second_order_unary(array, log, grad_grad_op)
        # TODO(kshitij12345): Remove
        check_nth_order_unary(array, log, [grad_op, grad_grad_op], [1, 2])


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


@with_seed()
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
        # TODO(kshitij12345): Remove
        check_nth_order_unary(array, sigmoid, [grad_op, grad_grad_op], [1, 2])
        check_nth_order_unary(array, sigmoid, grad_grad_op, 2)


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
    check_nth_order_unary(x, op, grad_grad_op, 2, rtol, atol)


def check_nth_order_unary(x, op, grad_ops, orders, rtol=None, atol=None):
    """Assert n-th order autograd gradient against expected gradient.

    Multiple order of gradients can be checked by passing list of
    function computing the particular order gradient and passing the
    corresponding list of order.

    Note
    ----
    1. Orders should always be monotonically increasing.
    2. Elements of grads_ops should correspond to elements of orders
    i.e. grads_op = [grad_op, grad_grad_grad_op] should be passed with
         orders = [1, 3]

    Parameters
    ----------
    x : mxnet.NDArray
        Input Array.
    op : Callable
        Operation to perform on Input Array.
    grad_ops : Callable or List of Callable
        Function to compute and assert gradient of given order.
    orders : int or List of int
        Order/s to assert expected and computed gradients.

    Returns
    -------
    None

    """
    if isinstance(orders, int):
        orders = [orders]
        grad_ops = [grad_ops]

    assert all(i < j for i, j in zip(orders[0:-1], orders[1:])), \
        "orders should be monotonically increasing"
    assert len(set(orders)) == len(orders), \
        "orders should have unique elements"
    highest_order = max(orders)

    x = nd.array(x)
    x.attach_grad()

    expected_grads = [grad_op(x) for grad_op in grad_ops]
    computed_grads = []
    head_grads = []

    # Perform compute.
    with autograd.record():
        y = op(x)
        for current_order in range(1, highest_order+1):
            head_grad = nd.random.normal(shape=x.shape)
            y = autograd.grad(heads=y, variables=x, head_grads=head_grad,
                              create_graph=True, retain_graph=True)[0]
            if current_order in orders:
                computed_grads.append(y)
            head_grads.append(head_grad)

    # Validate all the gradients.
    for order, grad, computed_grad in \
            zip(orders, expected_grads, computed_grads):
        # Compute expected values.
        expected_grad = grad.asnumpy()
        for head_grad in head_grads[:order]:
            expected_grad *= head_grad.asnumpy()

        assert_almost_equal(
            expected_grad, computed_grad.asnumpy(), rtol=rtol, atol=atol)


if __name__ == '__main__':
    import nose
    nose.runmodule()
