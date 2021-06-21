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

"""Namespace for registering control flow ops for imperative programming."""

from ..ndarray import numpy_extension as _mx_nd_npx
from ..util import set_module


__all__ = ["foreach", "while_loop", "cond"]


@set_module('mxnet.numpy_extension')
def foreach(body, data, init_states):
    """Run a for loop with user-defined computation over NDArrays on dimension 0.

    This operator simulates a for loop and body has the computation for an iteration
    of the for loop. It runs the computation in body on each slice from the input
    NDArrays.

    body takes two arguments as input and outputs a tuple of two elements,
    as illustrated below::

        out, states = body(data1, states)

    data1 can be either an NDArray or a list of NDArrays. If data is an NDArray,
    data1 is an NDArray. Otherwise, data1 is a list of NDArrays and has the same
    size as data. states is a list of NDArrays and have the same size as init_states.
    Similarly, out can be either an NDArray or a list of NDArrays, which are concatenated
    as the first output of foreach; states from the last execution of body
    are the second output of foreach.

    The computation done by this operator is equivalent to the pseudo code below
    when the input data is NDArray::

        states = init_states
        outs = []
        for i in data.shape[0]:
            s = data[i]
            out, states = body(s, states)
            outs.append(out)
        outs = stack(*outs)


    Parameters
    ----------
    body : HybridBlock.
        Define computation in an iteration.
    data: an NDArray or a list of NDArrays.
        The input data.
    init_states: an NDArray or nested lists of NDArrays.
        The initial values of the loop states.

    Returns
    -------
    outputs: an NDArray or nested lists of NDArrays.
        The output data concatenated from the output of all iterations.
    states: an NDArray or nested lists of NDArrays.
        The loop states in the last iteration.

    Examples
    --------
    >>> step = lambda data, states: (data + states[0], [states[0] * 2])
    >>> data = mx.np.random.uniform(size=(2, 10))
    >>> states = [mx.np.random.uniform(size=(10))]
    >>> outs, states = npx.control_flow.foreach(step, data, states)
    """
    return _mx_nd_npx.foreach(body, data, init_states)


#pylint: disable=W0621
@set_module('mxnet.numpy_extension')
def while_loop(cond, func, loop_vars, max_iterations=None):
    """Run a while loop with user-defined computation and loop condition.

    This operator simulates a while loop which iterately does customized computation
    as long as the condition is satisfied.

    `loop_vars` is a list of NDArrays on which the computation uses.

    `cond` is a user-defined function, used as the loop condition.
    It consumes `loop_vars`, and produces a scalar MXNet NDArray,
    indicating the termination of the loop.
    The loop ends when `cond` returns false (zero).
    The `cond` is variadic, and its signature should be
    `cond(*loop_vars) => NDArray`.

    `func` is a user-defined function, used as the loop body.
    It also consumes `loop_vars`, and produces `step_output` and `new_loop_vars` at each step.
    In each step, `step_output` should contain the same number elements.
    Through all steps, the i-th element of `step_output` should have the same shape and dtype.
    Also, `new_loop_vars` should contain the same number of elements as `loop_vars`,
    and the corresponding element should have the same shape and dtype.
    The `func` is variadic, and its signature should be
    `func(*loop_vars) =>
    (NDArray or nested List[NDArray] step_output, NDArray or nested List[NDArray] new_loop_vars)`.

    `max_iterations` is a scalar that defines the maximum number of iterations allowed.

    This function returns two lists.
    The first list has the length of `|step_output|`,
    in which the i-th element are all i-th elements of
    `step_output` from all steps, stacked along axis 0.
    The second list has the length of `|loop_vars|`,
    which represents final states of loop variables.

    .. warning::

       For now, the axis 0 of all NDArrays in the first list are `max_iterations`,
       due to lack of dynamic shape inference.

    .. warning::

       When `cond` is never satisfied, we assume `step_output` is empty,
       because it cannot be inferred. This is different from the symbolic version.

    Parameters
    ----------
    cond: a Python function.
        The loop condition.
    func: a Python function.
        The loop body.
    loop_vars: an NDArray or nested lists of NDArrays.
        The initial values of the loop variables.
    max_iterations: a python int.
        Maximum number of iterations.

    Returns
    ------
    outputs: an NDArray or nested lists of NDArrays
        stacked output from each step
    states: an NDArray or nested lists of NDArrays
        final state

    Examples
    --------
    >>> cond = lambda i, s: i <= 5
    >>> func = lambda i, s: ([i + s], [i + 1, s + i])
    >>> loop_vars = (mx.np.array([0], dtype="int64"), mx.np.array([1], dtype="int64"))
    >>> outputs, states = mx.npx.while_loop(cond, func, loop_vars, max_iterations=10)
    >>> outputs
    [array([[ 1],
           [ 2],
           [ 4],
           [ 7],
           [11],
           [16],
           [ 0],
           [ 0],
           [ 0],
           [ 0]], dtype=int64)]
    >>> states
    [array([6], dtype=int64), array([16], dtype=int64)]
    """
    return _mx_nd_npx.while_loop(cond, func, loop_vars, max_iterations=max_iterations)


@set_module('mxnet.numpy_extension')
def cond(pred, then_func, else_func, inputs, name="cond"):
    """Run an if-then-else using user-defined condition and computation

    This operator simulates a if-like branch which chooses to do one of
    the two customized computations according to the specified condition.

    `pred` is a scalar MXNet NDArray,
    indicating which branch of computation should be used.

    `then_func` is a user-defined function, used as computation of the then branch.
    It produces `outputs`, which is a list of NDArrays.
    The signature of `then_func` should be
    `then_func() => NDArray or nested List[NDArray]`.

    `else_func` is a user-defined function, used as computation of the else branch.
    It produces `outputs`, which is a list of NDArrays.
    The signature of `else_func` should be
    `else_func() => NDArray or nested List[NDArray]`.

    The `outputs` produces by `then_func` and `else_func` should have the same number
    of elements, all of which should be in the same shape, of the same dtype and stype.

    This function returns a list of symbols, representing the computation result.

    Parameters
    ----------
    pred: a Python function.
        The branch condition.
    then_func: a Python function.
        The computation to be executed if `pred` is true.
    else_func: a Python function.
        The computation to be executed if `pred` is false.

    Returns
    -------
    outputs: an NDArray or nested lists of NDArrays, representing the result of computation.

    Examples
    --------
    >>> a, b = mx.np.array([1]), mx.np.array([2])
    >>> pred = a * b < 5
    >>> then_func = lambda: (a + 5) * (b + 5)
    >>> else_func = lambda: (a - 5) * (b - 5)
    >>> outputs = mx.npx.cond(pred, then_func, else_func)
    >>> outputs[0]
    42.0
    """
    return _mx_nd_npx.cond(pred, then_func, else_func, inputs, name=name)
