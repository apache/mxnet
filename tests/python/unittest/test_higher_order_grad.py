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
from functools import reduce
from operator import mul
import random

from nose.tools import ok_

from common import with_seed
import mxnet
from mxnet import nd, autograd, gluon
from mxnet.test_utils import (
    assert_almost_equal, random_arrays, random_uniform_arrays, rand_shape_nd, same)


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
        return 1 - tanh(x)**2

    def grad_grad_op(x):
        return -2 * tanh(x) * grad_op(x)

    for dim in range(1, 5):
        shape = rand_shape_nd(dim)
        array = random_arrays(shape)
        check_nth_order_unary(array, tanh, grad_op, 1, rtol=1e-6, atol=1e-6)
        check_second_order_unary(
            array, tanh, grad_grad_op, rtol=1e-6, atol=1e-5)


@with_seed()
def test_arcsin():
    def arcsin(x):
        return nd.arcsin(x)

    def grad_grad_op(x):
        return x / nd.sqrt((1-x**2)**3)

    for dim in range(1, 5):
        shape = rand_shape_nd(dim)
        # Domain of arcsin is [-1, 1]
        array = random_uniform_arrays(shape, low=-0.99, high=0.99)[0]
        check_second_order_unary(array, arcsin, grad_grad_op)


@with_seed()
def test_arccos():
    def arccos(x):
        return nd.arccos(x)

    def grad_grad_op(x):
        return -x / nd.sqrt((1-x**2)**3)

    for dim in range(1, 5):
        shape = rand_shape_nd(dim)
        # Domain of arccos is [-1, 1]
        array = random_uniform_arrays(shape, low=-0.99, high=0.99)[0]
        check_second_order_unary(array, arccos, grad_grad_op)


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
def test_arcsinh():
    def arcsinh(x):
        return nd.arcsinh(x)

    def grad_grad_op(x):
        return x/nd.sqrt((nd.square(x)+1)**3)

    for dim in range(1, 5):
        shape = rand_shape_nd(dim)
        array = random_arrays(shape)
        check_second_order_unary(array, arcsinh, grad_grad_op)


@with_seed()
def test_arccosh():
    def arccosh(x):
        return nd.arccosh(x)

    def grad_grad_op(x):
        return x/(nd.sqrt(x-1) * nd.sqrt(x+1) * (x+1) * (x-1))

    sigma = random.randint(25, 100)
    mu = random.randint(500, 1000)

    for dim in range(1, 5):
        shape = rand_shape_nd(dim)
        array = random_arrays(shape)
        array = array * sigma + mu
        # Domain of arccosh 1 to infinity.
        assert((array > 1).all())
        check_second_order_unary(array, arccosh, grad_grad_op)


@with_seed()
def test_arctanh():
    def arctanh(x):
        return nd.arctanh(x)

    def grad_grad_op(x):
        return (2 * x)/((1 - x**2)**2)

    for dim in range(1, 5):
        shape = rand_shape_nd(dim)
        # Domain of arctanh is (-1, 1)
        array = random_uniform_arrays(shape, low=-0.99, high=0.99)[0]
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
def test_square():
    def grad_grad_op(x):
        return nd.ones_like(x) * 2

    for dim in range(1, 5):
        shape = rand_shape_nd(dim)
        array = random_arrays(shape)
        check_second_order_unary(array, nd.square, grad_grad_op)


@with_seed()
def test_expm1():
    def grad_grad_op(x):
        return nd.exp(x)

    for dim in range(1, 5):
        shape = rand_shape_nd(dim)
        array = random_arrays(shape)
        check_second_order_unary(array, nd.expm1, grad_grad_op)


@with_seed()
def test_log1p():
    def grad_grad_op(x):
        return -1/((1+x)**2)

    for dim in range(1, 5):
        shape = rand_shape_nd(dim)
        array = random_arrays(shape)
        check_second_order_unary(array, nd.log1p, grad_grad_op)


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


@with_seed()
def test_rsqrt():
    def rsqrt(x):
        return nd.rsqrt(x)

    def grad_grad_op(x):
        return 3/(4 * nd.sqrt(x**5))

    sigma = random.randint(25, 100)
    mu = random.randint(500, 1000)

    for dim in range(1, 5):
        shape = rand_shape_nd(dim)
        array = random_arrays(shape)
        array = sigma * array + mu
        # Only positive numbers
        assert((array > 0).all())
        check_second_order_unary(array, rsqrt, grad_grad_op)


@with_seed()
def test_rcbrt():
    def rcbrt(x):
        return nd.rcbrt(x)

    def grad_grad_op(x):
        return 4/(9 * nd.cbrt(x**7))

    sigma = random.randint(25, 100)
    mu = random.randint(500, 1000)

    for dim in range(1, 5):
        shape = rand_shape_nd(dim)
        array = random_arrays(shape)
        array = sigma * array + mu
        # Only positive numbers
        assert((array > 0).all())
        check_second_order_unary(array, rcbrt, grad_grad_op)


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

@with_seed()
def test_elemwise_sub():
    def sub(inputs):
        return nd.elemwise_sub(inputs[0], inputs[1])
    def grad_op(inputs):
        return  [nd.ones_like(inputs[0]), nd.negative(nd.ones_like(inputs[1]))]
    def grad_grad_op(inputs):
        return  [nd.zeros_like(inputs[0]),  nd.zeros_like(inputs[1])]

    for dim in range(1, 5):
        shape = rand_shape_nd(dim)
        x, y = random_arrays(shape, shape)
        check_nth_order_binary([x, y], sub, [grad_op, grad_grad_op], [1,  2])

@with_seed()
def test_elemwise_mul():
    def mul(inputs):
        return nd.elemwise_mul(inputs[0], inputs[1])
    def grad_op(inputs):
        return  [inputs[1], inputs[0]]
    def grad_grad_op(inputs):
        return [nd.zeros_like(inputs[0]) ,nd.zeros_like(inputs[1])]

    for dim in range(1, 5):
        shape = rand_shape_nd(dim)
        x, y = random_arrays(shape, shape)
        check_nth_order_binary([x, y], mul, [grad_op, grad_grad_op], [1,  2])

@with_seed()
def test_power():
    def power(inputs):
        return nd.power(inputs[0], inputs[1])

    def grad_op(inputs):
        x, y = inputs
        return  [y * nd.power(x, y - 1), nd.power(x, y) * nd.log(x)]

    def grad_grad_op(inputs):
        x, y = inputs
        return   [y * (y - 1) * nd.power(x, y - 2), nd.power(x, y) * (nd.log(x) ** 2)]

    def grad_grad_grad_op(inputs):
        x, y = inputs
        return   [y * (y - 1) * (y - 2) * nd.power(x, y - 3), nd.power(x, y) * (nd.log(x) ** 3)]

    low = 1.0
    high = 3.0
    for dim in range(1, 5):
        shape = rand_shape_nd(dim)
        x = nd.random.uniform(low, high, shape)
        y = nd.random.uniform(low, high, shape)
        check_nth_order_binary([x, y], power, [grad_op, grad_grad_op, grad_grad_grad_op], [1, 2, 3])

def autograd_grad_ex(heads, variables, head_grads=None, retain_graph=None, create_graph=False,
            train_mode=True):
    """ If some variables don't in the path of computing heads, we set the heads grad of them to zero
    instead of throwing exceptions.

    The autograd.grad requires user knows which variables involved to compute the heads grad of them.
    That's fine for first order grad, but for higher order grad, the variables used to compute the heads,
    may not used to computer their higher order grad. It's impossible to ask user to know
    the formulas of every order grad.

    E.g. we use such code to compute 2-nd order gradient:
      with autograd.record():
          z = op(x, y)
          head_grad = nd.ones_like(z)
          dz_dx, _  = autograd.grad(heads=z, variables=[x, y], head_grads=nd.ones_like(z),
                              create_graph=True, retain_graph=True)
          d2z_d2x, _  = autograd.grad(heads=dz_dx, variables=[x, y], head_grads=nd.ones_like(dz_dx),
                              create_graph=True, retain_graph=True)
    If z = x * y, because d2z_d2x = 0, MXNET will report the input is unreachable from the output.
    But it seems in that case MXNET returns zeros is more reasonable.
    """
    # xxx: only consider one head currently
    argument_names =  autograd.get_symbol(heads).list_arguments()

    # XXX: in some cases, a variable may has more than one outputs, we need a other way ot get  the name of various.
    # But in the unittest, it is fine
    variable_names = [autograd.get_symbol(variable).list_outputs()[0] for variable in variables]
    involved_variable_indexes = []
    involved_variables = []
    for i in range(0, len(variables)):
        if variable_names[i] in argument_names:
            involved_variables.append(variables[i])
            involved_variable_indexes.append(i)

    if involved_variables:
        partial_grads = autograd.grad(heads, involved_variables, head_grads, retain_graph, create_graph, train_mode)
    else:
        partial_grads = []

    grads = []
    partial_grads_index = 0
    for i in range(0, len(variables)):
       if i in involved_variable_indexes:
           grads.append(partial_grads[partial_grads_index])
           partial_grads_index += 1
       else:
           grads.append(nd.zeros_like(variables[i]))
    return grads


def check_nth_order_binary(inputs, op, grad_ops, orders, rtol=None, atol=None):
    """Assert n-th order autograd gradient against expected gradient.

    Multiple order of gradients can be checked by passing list of
    function computing the particular order gradient and passing the corresponding list of order.
    Note
    ----
    1. Orders should always be monotonically increasing.
    2. Elements of grads_ops should correspond to elements of orders
    i.e. grads_op = [grad_op, grad_grad_grad_op] should be passed with
         orders = [1, 3]

    Parameters
    ----------
    inputs : tuple of mxnet.NDArray (x, y)
        Input Array.
    op : Callable (x,y) -> z
        Operation to perform on Input Array.
    grad_ops : Callable or List of Callable
        Function (x,y) -> (n_grad_x, n_grad_y) to compute and assert gradient of given order.
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

    inputs = [nd.array(input) for input in inputs]
    for input in inputs:
        input.attach_grad()

    expected_grads = [grad_op(inputs) for grad_op in grad_ops]
    computed_grads = []
    head_grads = [[]]

    # Perform compute.
    with autograd.record():
        z = op(inputs)
        heads = [z for _ in inputs]
        for current_order in range(1, highest_order+1):
            grads = []
            new_head_grads = []
            new_heads = []
            for i in range(0, len(heads)):
                head = heads[i]
                head_grad = nd.random.normal(shape=head.shape)
                new_head_grads.append(head_grad.asnumpy())
                grads.append(autograd_grad_ex(heads=head, variables=inputs, head_grads=head_grad,
                                              create_graph=True, retain_graph=True)[i])
                # If we only use once auto grad with head_grads = head_grad in every iteration,
                # in the i-th iteration, we use head = derivative_(i-1) * head_grad_(i-1)
                # but in the expected computed, we use  head = derivative_(i-1)
                # why it is woks in the check_nth_order_unary?
                # Because most of them defined gradient of the first gradient (derivative_(1) * head_grad_(1))
                # of the function, and in the gradient function, they manually defined derivative_(i-1)
                # and use it to computed derivative_(1) * head_grad_(1).
                # It maybe a wrong approach, because the gradient of the first gradient should compute the grad of
                # derivative_(1) * head_grad_(1) instead of gradient of derivative_(1).
                new_heads.append(autograd_grad_ex(heads=head, variables=inputs, head_grads=nd.ones_like(head),
                                              create_graph=True, retain_graph=True)[i])
            heads = new_heads
            if current_order in orders:
                computed_grads.append(grads)
            head_grads.append(new_head_grads)

    # Validate all the gradients.
    for order, grad_list, computed_grad_list in \
            zip(orders, expected_grads, computed_grads):
        # Compute expected values.
        # keep as numpy value and use dot mul
        expected_grad_list = [grad.asnumpy() for grad in grad_list]
        for expected_grad, head_grad, computed_grad in zip(expected_grad_list, head_grads[order], computed_grad_list):
            expected_grad *= head_grad
            assert_almost_equal(expected_grad, computed_grad.asnumpy(), rtol=rtol, atol=atol)

def arange_shape_like(y):
    shape = y.shape
    nelems = reduce(mul, shape)
    x = nd.arange(nelems).reshape(shape)
    return x


class NDArrayGenerator(object):
    def __init__(self, dim, startdim=1):
        self.dim = dim
        self.curdim = startdim

    def __iter__(self):
        return self

    @staticmethod
    def gen(dimensions):
        shape = rand_shape_nd(dimensions, 4)
        nelems = reduce(mul, shape)
        x = nd.arange(nelems).reshape(shape)
        return x

    def next(self):
        return self.__next__()

    def __next__(self):
        if self.curdim > self.dim:
            raise StopIteration
        x = NDArrayGenerator.gen(self.curdim)
        self.curdim += 1
        return x


def flatten2d_right(x):
    s_0 = x.shape[0]
    s_1 = reduce(mul, x.shape[1:])
    return x.reshape((s_0, s_1))


def flatten2d_left(x):
    s_0 = reduce(mul, x.shape[:-1])
    s_1 = x.shape[-1]
    return x.reshape((s_0, s_1))


@with_seed()
def test_dense_backward_flatten():
    print("2nd order gradient for Fully Connected, flatten=True")
    for x in NDArrayGenerator(4,2):
        hidden = random.randrange(1, 4)
        net = gluon.nn.Sequential()
        with net.name_scope():
            net.add(gluon.nn.Dense(hidden, flatten=True))
        net.initialize(mxnet.initializer.Constant(.5))
        x.attach_grad()
        with autograd.record():
            y = net.forward(x)
            o_y = arange_shape_like(y)  # head gradient of y
            params = [p.data() for p in net.collect_params().values()]
            w = params[0]
            b = params[1]
            print("Checking y ({}) = x({}) * w^T({}) + b({})".format(y.shape, x.shape, w.shape, b.shape))
            x_grad = autograd.grad(heads=y, variables=x, head_grads=o_y,
                                   create_graph=True, retain_graph=True)[0]
            o_x_grad = arange_shape_like(x_grad)
            w_grad_grad = autograd.grad(heads=x_grad, variables=w,
                                        head_grads=o_x_grad, create_graph=False)[0]
            w_grad = autograd.grad(heads=y, variables=w, head_grads=o_y,
                                   create_graph=True, retain_graph=True)[0]
            o_w_grad = arange_shape_like(w_grad)
            x_grad_grad = autograd.grad(heads=w_grad, variables=x,
                                        head_grads=o_w_grad, create_graph=False)[0]
        # Expected results
        w_grad_e = nd.dot(o_y, x, transpose_a=True)
        w_grad_grad_e = nd.dot(o_y, o_x_grad, transpose_a=True)
        x_grad_e = nd.dot(o_y, w)
        x_grad_grad_e = nd.dot(o_y, o_w_grad)
        ok_(w_grad.shape == w.shape)
        ok_(w_grad_grad.shape == w.shape)
        ok_(x_grad.shape == x.shape)
        ok_(x_grad_grad.shape == x.shape)
        w_grad_check = same(flatten2d_right(w_grad), flatten2d_right(w_grad_e))
        w_grad_grad_check = same(flatten2d_right(w_grad_grad), flatten2d_right(w_grad_grad_e))
        x_grad_check = same(flatten2d_right(x_grad), flatten2d_right(x_grad_e))
        x_grad_grad_check = same(flatten2d_right(x_grad_grad), flatten2d_right(x_grad_grad_e))
        ok_(x_grad_check)
        ok_(w_grad_check)
        ok_(x_grad_grad_check)
        ok_(w_grad_grad_check)

@with_seed()
def test_dense_backward_no_flatten():
    print("2nd order gradient for Fully Connected, flatten=False")
    for x in NDArrayGenerator(5,3):
        hidden = random.randrange(1, 4)
        net = gluon.nn.Sequential()
        with net.name_scope():
            net.add(gluon.nn.Dense(hidden, flatten=False))
        net.initialize(mxnet.initializer.Constant(.5))
        x.attach_grad()
        with autograd.record():
            y = net.forward(x)
            o_y = arange_shape_like(y)  # head gradient of y
            params = [p.data() for p in net.collect_params().values()]
            w = params[0]
            b = params[1]
            print("Checking y ({}) = x({}) * w^T({}) + b({})".format(y.shape, x.shape, w.shape, b.shape))
            x_grad = autograd.grad(heads=y, variables=x, head_grads=o_y,
                                   create_graph=True, retain_graph=True)[0]
            o_x_grad = arange_shape_like(x_grad)
            w_grad_grad = autograd.grad(heads=x_grad, variables=w,
                                        head_grads=o_x_grad, create_graph=False)[0]
            w_grad = autograd.grad(heads=y, variables=w, head_grads=o_y,
                                   create_graph=True, retain_graph=True)[0]
            o_w_grad = arange_shape_like(w_grad)
            x_grad_grad = autograd.grad(heads=w_grad, variables=x,
                                        head_grads=o_w_grad, create_graph=False)[0]
        # Expected results
        o_y = flatten2d_left(o_y)
        x = flatten2d_left(x)
        o_x_grad = flatten2d_left(o_x_grad)
        o_w_grad = flatten2d_left(o_w_grad)
        w_grad_e = nd.dot(o_y, x, transpose_a=True)
        w_grad_grad_e = nd.dot(o_y, o_x_grad, transpose_a=True)
        x_grad_e = nd.dot(o_y, w)
        x_grad_grad_e = nd.dot(o_y, o_w_grad)
        w_grad_check = same(flatten2d_left(w_grad), flatten2d_left(w_grad_e))
        w_grad_grad_check = same(flatten2d_left(w_grad_grad), flatten2d_left(w_grad_grad_e))
        x_grad_check = same(flatten2d_left(x_grad), flatten2d_left(x_grad_e))
        x_grad_grad_check = same(flatten2d_left(x_grad_grad), flatten2d_left(x_grad_grad_e))
        ok_(x_grad_check)
        ok_(w_grad_check)
        ok_(x_grad_grad_check)
        ok_(w_grad_grad_check)


if __name__ == '__main__':
    import nose
    nose.runmodule()
