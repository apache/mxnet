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

# coding: utf-8
# pylint: disable=wildcard-import, unused-wildcard-import,redefined-outer-name
"""Contrib NDArray API of MXNet."""
import math
import numpy as np
import mxnet as mx
from ..device import current_device
from ..random import uniform
from ..base import _as_list
from . import ndarray
try:
    from .gen_contrib import *
except ImportError:
    pass

__all__ = ["rand_zipfian", "foreach", "while_loop", "cond", "isinf", "isfinite", "isnan"]

def _flatten_list(nested_list):
    return [item for sublist in nested_list for item in sublist]

# pylint: disable=line-too-long
def rand_zipfian(true_classes, num_sampled, range_max, ctx=None):
    """Draw random samples from an approximately log-uniform or Zipfian distribution.

    This operation randomly samples *num_sampled* candidates the range of integers [0, range_max).
    The elements of sampled_candidates are drawn with replacement from the base distribution.

    The base distribution for this operator is an approximately log-uniform or Zipfian distribution:

    P(class) = (log(class + 2) - log(class + 1)) / log(range_max + 1)

    This sampler is useful when the true classes approximately follow such a distribution.
    For example, if the classes represent words in a lexicon sorted in decreasing order of \
    frequency. If your classes are not ordered by decreasing frequency, do not use this op.

    Additionaly, it also returns the number of times each of the \
    true classes and the sampled classes is expected to occur.

    Parameters
    ----------
    true_classes : NDArray
        A 1-D NDArray of the target classes.
    num_sampled: int
        The number of classes to randomly sample.
    range_max: int
        The number of possible classes.
    ctx : Context
        Device context of output. Default is current context.

    Returns
    -------
    samples: NDArray
        The sampled candidate classes in 1-D `int64` dtype.
    expected_count_true: NDArray
        The expected count for true classes in 1-D `float64` dtype.
    expected_count_sample: NDArray
        The expected count for sampled candidates in 1-D `float64` dtype.

    Examples
    --------
    >>> true_cls = mx.nd.array([3])
    >>> samples, exp_count_true, exp_count_sample = mx.nd.contrib.rand_zipfian(true_cls, 4, 5)
    >>> samples
    [1 3 3 3]
    <NDArray 4 @cpu(0)>
    >>> exp_count_true
    [ 0.12453879]
    <NDArray 1 @cpu(0)>
    >>> exp_count_sample
    [ 0.22629439  0.12453879  0.12453879  0.12453879]
    <NDArray 4 @cpu(0)>
    """
    if ctx is None:
        ctx = current_device()
    log_range = math.log(range_max + 1)
    rand = uniform(0, log_range, shape=(num_sampled,), dtype='float64', ctx=ctx)
    # make sure sampled_classes are in the range of [0, range_max)
    sampled_classes = (rand.exp() - 1).astype('int64') % range_max

    true_cls = true_classes.as_in_context(ctx).astype('float64')
    expected_count_true = ((true_cls + 2.0) / (true_cls + 1.0)).log() / log_range * num_sampled
    # cast sampled classes to fp64 to avoid interget division
    sampled_cls_fp64 = sampled_classes.astype('float64')
    expected_prob_sampled = ((sampled_cls_fp64 + 2.0) / (sampled_cls_fp64 + 1.0)).log() / log_range
    expected_count_sampled = expected_prob_sampled * num_sampled
    return sampled_classes, expected_count_true, expected_count_sampled
# pylint: enable=line-too-long


def _flatten(args, inout_str):
    if isinstance(args, ndarray.NDArray):
        return [args], int(0)

    assert isinstance(args, (list, tuple)), \
        f"{inout_str} must be (nested) list of NDArray, " \
        f"but got {str(args)} of type {str(type(args))}"
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
        f"but got {str(args)} of type {str(type(args))}"
    ret = []
    for i in fmt:
        res, args = _regroup(args, i)
        ret.append(res)
    return ret, args


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
    >>> data = mx.nd.random.uniform(shape=(2, 10))
    >>> states = [mx.nd.random.uniform(shape=(10))]
    >>> outs, states = mx.nd.contrib.foreach(step, data, states)
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
    check_input(flatten, ndarray.NDArray,
                "data should be an NDArray or a nested list of NDArrays")
    flatten, _ = _flatten(init_states, "foreach states")
    check_input(flatten, ndarray.NDArray,
                "init_states should be an NDArray or a nested list of NDArrays")

    not_data_list = isinstance(data, ndarray.NDArray)
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
        tmp_outputs.append(ndarray.op.stack(*out))
    outputs = tmp_outputs
    outputs, _ = _regroup(outputs, out_fmt)

    return (outputs, states)

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
    >>> loop_vars = (mx.nd.array([0], dtype="int64"), mx.nd.array([1], dtype="int64"))
    >>> outputs, states = mx.nd.contrib.while_loop(cond, func, loop_vars, max_iterations=10)
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
        if isinstance(inputs, ndarray.NDArray):
            inputs = inputs.asscalar()
        try:
            inputs = type_(inputs)
        except:
            raise ValueError(f"Cannot convert {name} to python {type_.__name__}")
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
    not_loop_var_list = isinstance(loop_vars, ndarray.NDArray)
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
        items = [x.expand_dims(0) for x in items]
        if steps != max_iterations and items:
            pad_shape = [max_iterations - steps] + list(items[0].shape[1: ])
            pad = ndarray.empty(
                shape=pad_shape,
                ctx=items[0].context,
                dtype=items[0].dtype,
            )
            items = list(items) + [pad]
        try:
            stacked_outputs.append(ndarray.op.concat(*items, dim=0))
        except ValueError:
            raise ValueError("\n".join(
                [f"Shapes of {i_th}-th elements in step_outputs are inconsistent, which are:"] +
                [f"  Step {i}, shape is {str(x.shape)}" for i, x in enumerate(items)]
            ))
    if out_fmt is not None:
        stacked_outputs, _ = _regroup(stacked_outputs, out_fmt)
    if not_loop_var_list:
        loop_vars = loop_vars[0]
    return stacked_outputs, loop_vars

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
            inputs = inputs.asscalar()
        try:
            inputs = type_(inputs)
        except:
            raise ValueError(f"Cannot convert {name} to python {type_.__name__}")
        return inputs

    branch = _to_python_scalar(pred, bool, "pred")
    if branch:
        return then_func()
    else:
        return else_func()

def isinf(data):
    """Performs an element-wise check to determine if the NDArray contains an infinite element
    or not.


    Parameters
    ----------
    input : NDArray
        An N-D NDArray.

    Returns
    -------
    output: NDArray
        The output NDarray, with same shape as input, where 1 indicates the array element is
        equal to positive or negative infinity and 0 otherwise.

    Examples
    --------
    >>> data = mx.nd.array([np.inf, -np.inf, np.NINF, -1])
    >>> output = mx.nd.contrib.isinf(data)
    >>> output
    [1. 1. 1. 0.]
    <NDArray 4 @cpu(0)>
    """
    return data.abs() == np.inf

def isfinite(data):
    """Performs an element-wise check to determine if the NDArray contains an infinite element
    or not.


    Parameters
    ----------
    input : NDArray
        An N-D NDArray.

    Returns
    -------
    output: NDArray
        The output NDarray, with same shape as input, where 1 indicates the array element is
        finite i.e. not equal to positive or negative infinity and 0 in places where it is
        positive or negative infinity.

    Examples
    --------
    >>> data = mx.nd.array([np.inf, -np.inf, np.NINF, -1])
    >>> output = mx.nd.contrib.isfinite(data)
    >>> output
    [0. 0. 0. 1.]
    <NDArray 4 @cpu(0)>
    """
    is_data_not_nan = data == data  # pylint: disable=comparison-with-itself
    is_data_not_infinite = data.abs() != np.inf
    return ndarray.logical_and(is_data_not_infinite, is_data_not_nan)

def isnan(data):
    """Performs an element-wise check to determine if the NDArray contains a NaN element
    or not.


    Parameters
    ----------
    data : NDArray
        An N-D NDArray.

    Returns
    -------
    output: NDArray
        The output NDarray, with same shape as input, where 1 indicates the array element is
        NaN i.e. Not a Number and 0 otherwise.

    Examples
    --------
    >>> data = mx.nd.array([np.nan, -1])
    >>> output = mx.nd.contrib.isnan(data)
    >>> output
    [1. 0.]
    <NDArray 2 @cpu(0)>
    """
    return data != data  # pylint: disable=comparison-with-itself

def _get_rescale_grad(rescale_grad, ctx=mx.cpu()):
    if not isinstance(rescale_grad, ndarray.NDArray):
        return ndarray.full(shape=(1,), val=rescale_grad, ctx=ctx)
    else:
        return rescale_grad.as_in_context(ctx)

def adamw_update(weight, grad, mean, var, rescale_grad, lr, eta, beta1=0.9, beta2=0.999,
                 epsilon=1e-8, wd=0, clip_gradient=-1, out=None, name=None, **kwargs):
    rescale_grad = _get_rescale_grad(rescale_grad, ctx=weight.context)
    return ndarray._internal._adamw_update(weight=weight, grad=grad, mean=mean, var=var,
                                           rescale_grad=rescale_grad, lr=lr, eta=eta,
                                           beta1=beta1, beta2=beta2, epsilon=epsilon,
                                           wd=wd, clip_gradient=clip_gradient, out=out,
                                           name=name, **kwargs)

def mp_adamw_update(weight, grad, mean, var, weight32, rescale_grad, lr, eta, beta1=0.9,
                    beta2=0.999, epsilon=1e-8, wd=0, clip_gradient=-1, out=None,
                    name=None, **kwargs):
    rescale_grad = _get_rescale_grad(rescale_grad, ctx=weight.context)
    return ndarray._internal._mp_adamw_update(weight=weight, grad=grad, mean=mean, var=var,
                                              weight32=weight32,
                                              rescale_grad=rescale_grad, lr=lr, eta=eta,
                                              beta1=beta1, beta2=beta2, epsilon=epsilon,
                                              wd=wd, clip_gradient=clip_gradient, out=out,
                                              name=name, **kwargs)

def multi_adamw_update(weights, grads, mean, var, rescale_grad, lrs, wds, etas,
                       out=None, name=None, size=0, **kwargs):
    if not size:
        size = len(weights)

    rescale_grad = _get_rescale_grad(rescale_grad, ctx=weights[0].context)
    temp_list = _flatten_list(zip(weights, grads, mean, var)) + [rescale_grad]
    return ndarray._internal._multi_adamw_update(*temp_list,
                                                 out=out,
                                                 num_weights=size,
                                                 lrs=lrs,
                                                 wds=wds,
                                                 etas=etas,
                                                 name=name,
                                                 **kwargs)

def multi_mp_adamw_update(weights, grads, mean, var, weights32, rescale_grad, lrs, wds, etas,
                          out=None, name=None, size=0, **kwargs):
    if not size:
        size = len(weights)

    rescale_grad = _get_rescale_grad(rescale_grad, ctx=weights[0].context)
    temp_list = _flatten_list(zip(weights, grads, mean, var, weights32)) + [rescale_grad]
    return ndarray._internal._multi_mp_adamw_update(*temp_list,
                                                    out=out,
                                                    num_weights=size,
                                                    lrs=lrs,
                                                    wds=wds,
                                                    etas=etas,
                                                    name=name,
                                                    **kwargs)

def multi_lamb_update(weights, grads, mean, var, step_count,
                      lrs, wds, out=None, num_tensors=0, **kwargs):
    """Given a list of gradients, update weights, mean and variance of multiple tensors
    following LAMB Optimizer implementation.

    Parameters
    ----------
    weights : List of NDArrays containing the input weights of multiple tensors

    grads : List of NDArrays containing input gradients

    mean : List of NDArrays containing mean of multiple tensors to be updated

    var : List of NDArrays containing variance of multiple tensors to be updated

    step_count : List of scalars with the number of update step for each tensor

    lrs : List of learning rates (one for each tensor)

    wds : List of weight decays (one for each tensor)

    out: List of NDArrays where the updated weights will be stored

    num_tensors : Number of NDArrays/tensors in the list
    """

    if not num_tensors:
        num_tensors = len(weights)
    temp_list = _flatten_list(zip(weights, grads, mean, var))
    return ndarray._internal._multi_lamb_update(*temp_list,
                                                out=out,
                                                num_tensors=num_tensors,
                                                step_count=step_count,
                                                learning_rates=lrs,
                                                wds=wds,
                                                **kwargs)

def multi_mp_lamb_update(weights, grads, mean, var, weights32, step_count,
                         lrs, wds, out=None, num_tensors=0, **kwargs):
    """Given a list of gradients, update weights, mean and variance of multiple tensors
    following LAMB Optimizer implementation, and using Mixed-Precision.

    Parameters
    ----------
    weights : List of NDArrays containing the input weights of multiple tensors

    grads : List of NDArrays containing input gradients

    mean : List of NDArrays containing mean of multiple tensors to be updated

    var : List of NDArrays containing variance of multiple tensors to be updated

    weights32 : Master copy of weights in FP32

    step_count : List of scalars with the number of update step for each tensor

    lrs : List of learning rates (one for each tensor)

    wds : List of weight decays (one for each tensor)

    out: List of NDArrays where the updated weights will be stored

    num_tensors : Number of NDArrays/tensors in the list
    """

    if not num_tensors:
        num_tensors = len(weights)
    temp_list = _flatten_list(zip(weights, grads, mean, var, weights32))
    return ndarray._internal._multi_mp_lamb_update(*temp_list,
                                                   out=out,
                                                   num_tensors=num_tensors,
                                                   step_count=step_count,
                                                   learning_rates=lrs,
                                                   wds=wds,
                                                   **kwargs)

def adabelief_update(weight, grad, mean, var, rescale_grad, lr, eta, beta1=0.9, beta2=0.999,
                     epsilon=1e-8, wd=0, clip_gradient=-1, out=None, name=None, **kwargs):
    rescale_grad = _get_rescale_grad(rescale_grad, ctx=weight.context)
    return ndarray._internal._adabelief_update(weight=weight, grad=grad, mean=mean, var=var,
                                               rescale_grad=rescale_grad, lr=lr, eta=eta,
                                               beta1=beta1, beta2=beta2, epsilon=epsilon,
                                               wd=wd, clip_gradient=clip_gradient, out=out,
                                               name=name, **kwargs)

def mp_adabelief_update(weight, grad, mean, var, weight32, rescale_grad, lr, eta, beta1=0.9,
                        beta2=0.999, epsilon=1e-8, wd=0, clip_gradient=-1, out=None,
                        name=None, **kwargs):
    rescale_grad = _get_rescale_grad(rescale_grad, ctx=weight.context)
    return ndarray._internal._mp_adabelief_update(weight=weight, grad=grad, mean=mean, var=var,
                                                  weight32=weight32,
                                                  rescale_grad=rescale_grad, lr=lr, eta=eta,
                                                  beta1=beta1, beta2=beta2, epsilon=epsilon,
                                                  wd=wd, clip_gradient=clip_gradient, out=out,
                                                  name=name, **kwargs)

def multi_adabelief_update(weights, grads, mean, var, rescale_grad, lrs, wds, etas,
                           out=None, name=None, size=0, **kwargs):
    if not size:
        size = len(weights)

    rescale_grad = _get_rescale_grad(rescale_grad, ctx=weights[0].context)
    temp_list = _flatten_list(zip(weights, grads, mean, var)) + [rescale_grad]
    return ndarray._internal._multi_adabelief_update(*temp_list,
                                                     out=out,
                                                     num_weights=size,
                                                     lrs=lrs,
                                                     wds=wds,
                                                     etas=etas,
                                                     name=name,
                                                     **kwargs)

def multi_mp_adabelief_update(weights, grads, mean, var, weights32, rescale_grad, lrs, wds, etas,
                              out=None, name=None, size=0, **kwargs):
    if not size:
        size = len(weights)

    rescale_grad = _get_rescale_grad(rescale_grad, ctx=weights[0].context)
    temp_list = _flatten_list(zip(weights, grads, mean, var, weights32)) + [rescale_grad]
    return ndarray._internal._multi_mp_adabelief_update(*temp_list,
                                                        out=out,
                                                        num_weights=size,
                                                        lrs=lrs,
                                                        wds=wds,
                                                        etas=etas,
                                                        name=name,
                                                        **kwargs)

def multi_lans_update(weights, grads, mean, var, step_count,
                      lrs, wds, out=None, num_tensors=0, **kwargs):
    """Given a list of gradients, update weights, mean and variance of multiple tensors
    following LANS Optimizer implementation.

    Parameters
    ----------
    weights : List of NDArrays containing the input weights of multiple tensors

    grads : List of NDArrays containing input gradients

    mean : List of NDArrays containing mean of multiple tensors to be updated

    var : List of NDArrays containing variance of multiple tensors to be updated

    step_count : List of scalars with the number of update step for each tensor

    lrs : List of learning rates (one for each tensor)

    wds : List of weight decays (one for each tensor)

    out: List of NDArrays where the updated weights will be stored

    num_tensors : Number of NDArrays/tensors in the list
    """

    if not num_tensors:
        num_tensors = len(weights)
    temp_list = _flatten_list(zip(weights, grads, mean, var))
    return ndarray._internal._multi_lans_update(*temp_list,
                                                out=out,
                                                num_tensors=num_tensors,
                                                step_count=step_count,
                                                learning_rates=lrs,
                                                wds=wds,
                                                **kwargs)


def multi_mp_lans_update(weights, grads, mean, var, weights32, step_count,
                         lrs, wds, out=None, num_tensors=0, **kwargs):
    """Given a list of gradients, update weights, mean and variance of multiple tensors
    following LANS Optimizer implementation, and using Mixed-Precision.

    Parameters
    ----------
    weights : List of NDArrays containing the input weights of multiple tensors

    grads : List of NDArrays containing input gradients

    mean : List of NDArrays containing mean of multiple tensors to be updated

    var : List of NDArrays containing variance of multiple tensors to be updated

    weights32 : Master copy of weights in FP32

    step_count : List of scalars with the number of update step for each tensor

    lrs : List of learning rates (one for each tensor)

    wds : List of weight decays (one for each tensor)

    out: List of NDArrays where the updated weights will be stored

    num_tensors : Number of NDArrays/tensors in the list
    """

    if not num_tensors:
        num_tensors = len(weights)
    temp_list = _flatten_list(zip(weights, grads, mean, var, weights32))
    return ndarray._internal._multi_mp_lans_update(*temp_list,
                                                   out=out,
                                                   num_tensors=num_tensors,
                                                   step_count=step_count,
                                                   learning_rates=lrs,
                                                   wds=wds,
                                                   **kwargs)
