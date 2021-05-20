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

"""Namespace for registering numpy_extension ops for imperative programming."""

from ..util import set_module
from ..ndarray import NDArray
from ..base import _as_list
from .. import numpy as _mx_np


__all__ = ["foreach", "while_loop", "cond"]


def _flatten(args, inout_str):
    if isinstance(args, NDArray):
        return [args], int(0)

    assert isinstance(args, (list, tuple)), \
        "%s must be (nested) list of NDArray, " \
        "but got %s of type %s"%(inout_str, str(args), str(type(args)))
    flat = []
    fmts = []
    for i in args:
        arg, fmt = _flatten(i, inout_str)
        flat.extend(arg)
        fmts.append(fmt)
    return flat, fmts


def _regroup(args, fmt):
    if isinstance(fmt, int):
        if fmt == 0:
            return args[0], args[1:]
        return args[:fmt], args[fmt:]

    assert isinstance(args, (list, tuple)), \
        "output must be (nested) list of NDArray, " \
        "but got %s of type %s"%(str(args), str(type(args)))
    ret = []
    for i in fmt:
        res, args = _regroup(args, i)
        ret.append(res)
    return ret, args


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
    body : a Python function.
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

    def check_input(inputs, in_type, msg):
        is_NDArray_or_list = True
        if isinstance(inputs, list):
            for i in inputs:
                if not isinstance(i, in_type):
                    is_NDArray_or_list = False
                    break
        else:
            is_NDArray_or_list = isinstance(inputs, in_type)
        assert is_NDArray_or_list, msg

    flatten, _ = _flatten(data, "foreach input")
    check_input(flatten, NDArray,
                "data should be an NDArray or a nested list of NDArrays")
    flatten, _ = _flatten(init_states, "foreach states")
    check_input(flatten, NDArray,
                "init_states should be an NDArray or a nested list of NDArrays")

    not_data_list = isinstance(data, NDArray)
    num_iters = data.shape[0] if not_data_list else data[0].shape[0]
    states = init_states
    outputs = []
    for i in range(num_iters):
        if not_data_list:
            eles = data[i]
        else:
            eles = [d[i] for d in data]
        outs, states = body(eles, states)
        outs, out_fmt = _flatten(outs, "foreach output")
        outputs.append(outs)
    outputs = zip(*outputs)
    tmp_outputs = []
    for out in outputs:
        tmp_outputs.append(_mx_np.stack(out))
    outputs = tmp_outputs
    outputs, _ = _regroup(outputs, out_fmt)

    return (outputs, states)


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
    [
    [[ 1]
    [ 2]
    [ 4]
    [ 7]
    [11]
    [16]
    [...]  # undefined value
    [...]
    [...]
    [...]]
    <NDArray 6x1 @cpu(0)>]
    >>> states
    [
    [6]
    <NDArray 1 @cpu(0)>,
    [16]
    <NDArray 1 @cpu(0)>]
    """
    def _to_python_scalar(inputs, type_, name):
        """Converts "inputs", possibly typed mxnet NDArray, a numpy ndarray, other python types,
        to the given type
        """
        if isinstance(inputs, NDArray):
            inputs = inputs.item()
        try:
            inputs = type_(inputs)
        except:
            raise ValueError("Cannot convert %s to python %s" % (name, type_.__name__))
        return inputs

    def _func_wrapper(loop_vars):
        """This wrapper unifies
             "func: loop_vars -> new_loop_vars"
         and "func: loop_vars -> (step_output, new_loop_vars)"
        into "func: loop_vars -> (None or tuple of step_outputs, tuple of new_loop_vars)
        """
        step_output, new_loop_vars = func(*loop_vars)
        if step_output is None:
            step_output = []
        if new_loop_vars is None:
            new_loop_vars = []
        if isinstance(step_output, tuple):
            step_output = list(step_output)
        if isinstance(new_loop_vars, tuple):
            new_loop_vars = list(new_loop_vars)
        new_loop_vars = _as_list(new_loop_vars)
        if len(loop_vars) != len(new_loop_vars):
            raise ValueError("The length of loop_vars should be consistent during the loop")
        return step_output, new_loop_vars

    if max_iterations is None:
        raise ValueError("max_iterations should be specified")
    max_iterations = _to_python_scalar(max_iterations, int, "max_iteration")
    # It should be work as fine if loop_vars are empty I guess,
    # but it is semantically unnecessary to include this case.
    if len(loop_vars) == 0:
        raise ValueError("loop_vars should contain at least one element")

    steps = 0
    outputs = []
    # there might not be an iteration.
    out_fmt = None
    not_loop_var_list = isinstance(loop_vars, NDArray)
    loop_vars = _as_list(loop_vars)
    while steps < max_iterations and \
            _to_python_scalar(cond(*loop_vars), bool, "Return value of cond"): # loop condition
        step_output, loop_vars = _func_wrapper(loop_vars)
        step_output, out_fmt = _flatten(step_output, "while output")
        outputs.append(step_output)
        steps += 1
        if len(outputs) != steps or len(step_output) != len(outputs[0]):
            raise ValueError("Number of elements in step_output should be the same in each step")
    stacked_outputs = []
    for i_th, items in enumerate(zip(*outputs), 1):
        # `mx.ndarray.pad` only support 4-D or 5-D inputs for now
        # so we could not use it.
        items = [_mx_np.expand_dims(x, 0) for x in items]
        try:
            concate_outputs = _mx_np.concatenate(items, axis=0)
            print(concate_outputs.shape)
            if steps != max_iterations and items:
                to_pad = max_iterations - steps
                concate_outputs = _mx_np.pad(concate_outputs, pad_width=((0, to_pad), (0, 0)))
            stacked_outputs.append(concate_outputs)
        except ValueError:
            raise ValueError("\n".join(
                ["Shapes of %d-th elements in step_outputs are inconsistent, which are:" % i_th] +
                ["  Step %d, shape is %s" % (i, str(x.shape)) for i, x in enumerate(items)]
            ))
    if out_fmt is not None:
        stacked_outputs, _ = _regroup(stacked_outputs, out_fmt)
    if not_loop_var_list:
        loop_vars = loop_vars[0]
    return stacked_outputs, loop_vars


@set_module('mxnet.numpy_extension')
def cond(pred, then_func, else_func):
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
    pred: a MXNet NDArray representing a scalar.
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
    >>> a, b = mx.nd.array([1]), mx.nd.array([2])
    >>> pred = a * b < 5
    >>> then_func = lambda: (a + 5) * (b + 5)
    >>> else_func = lambda: (a - 5) * (b - 5)
    >>> outputs = mx.nd.contrib.cond(pred, then_func, else_func)
    >>> outputs[0]
    [42.]
    <NDArray 1 @cpu(0)>
    """
    def _to_python_scalar(inputs, type_, name):
        """Converts "inputs", possibly typed mxnet NDArray, a numpy ndarray, other python types,
        to the given type
        """
        if hasattr(inputs, "asscalar"):
            inputs = inputs.item()
        try:
            inputs = type_(inputs)
        except:
            raise ValueError("Cannot convert %s to python %s" % (name, type_.__name__))
        return inputs

    branch = _to_python_scalar(pred, bool, "pred")
    if branch:
        return then_func()
    else:
        return else_func()
