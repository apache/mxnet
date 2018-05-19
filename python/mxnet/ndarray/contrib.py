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
# pylint: disable=wildcard-import, unused-wildcard-import
"""Contrib NDArray API of MXNet."""
import math
from ..context import current_context
from ..random import uniform
from ..base import _as_list
from .op import stack
try:
    from .gen_contrib import *
except ImportError:
    pass

__all__ = ["rand_zipfian"]

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
        ctx = current_context()
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

def foreach(func, data, init_states):
    """Run a for loop with user-defined computation over NDArrays on dimension 0.

    This operator simulates a for loop and func has the computation for an iteration
    of the for loop. It runs the computation in func on each slice from the input
    NDArrays.

    func takes two arguments as input and outputs a tuple of two elements,
    as illustrated below:

    out, states = func(data1, states)

    data1 can be either a symbol or a list of symbols. If data is a symbol,
    data1 is a symbol. Otherwise, data1 is a list of symbols and has the same
    size as data. states is a list of symbols and have the same size as init_states.
    Similarly, out can be either a symbol or a list of symbols, which are concatenated
    as the first output of foreach; states from the last execution of func
    are the second output of foreach.

    The computation done by this operator is equivalent to the pseudo code below
    when the input data is NDArray:

    states = init_states
    outs = []
    for i in data.shape[0]:
        s = data[i]
        out, states = func(s, states)
        outs.append(out)
    outs = stack(*outs)


    Parameters
    ----------
    func : a Python function.
        Define computation in an iteration.
    data: a symbol or a list of symbols.
        The input data.
    init_states: a list of symbols.
        The initial values of the loop states.
    name: string.
        The name of the operator.

    Returns
    -------
    outputs: a Symbol or a list of Symbols.
        The output data concatenated from the output of all iterations.
    states: a list of Symbols.
        The loop states in the last iteration.

    Examples
    --------
    >>> step = lambda data, states: (data + states[0], [states[0] * 2])
    >>> data = mx.nd.random.uniform(shape=(2, 10))
    >>> states = [mx.nd.random.uniform(shape=(10))]
    >>> outs, states = mx.nd.contrib.foreach(step, data, states)
    """

    assert isinstance(init_states, list), "init_states should be a list"
    states = init_states
    outputs = []
    for i in range(data.shape[0]):
        ele = data[i]
        outs, states = func(ele, states)
        outs = _as_list(outs)
        if i == 0:
            # outputs is a list of lists
            for out in outs:
                outputs.append([out])
        else:
            for j, out in enumerate(outs):
                outputs[j].append(out)
    for out in outputs:
        out = stack(*out)
    return (outputs, states)
