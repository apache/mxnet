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
"""Contrib Symbol API of MXNet."""
import math
import ctypes
import copy

from .random import uniform
from .symbol import Symbol
try:
    from .gen_contrib import *
except ImportError:
    pass

from . import symbol
from ..base import _LIB, check_call
from ..base import SymbolHandle, _as_list
from ..attribute import AttrScope

__all__ = ["rand_zipfian", "foreach"]

def rand_zipfian(true_classes, num_sampled, range_max):
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
    true_classes : Symbol
        The target classes in 1-D.
    num_sampled: int
        The number of classes to randomly sample.
    range_max: int
        The number of possible classes.

    Returns
    -------
    samples: Symbol
        The sampled candidate classes in 1-D `int64` dtype.
    expected_count_true: Symbol
        The expected count for true classes in 1-D `float64` dtype.
    expected_count_sample: Symbol
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
    assert(isinstance(true_classes, Symbol)), "unexpected type %s" % type(true_classes)
    log_range = math.log(range_max + 1)
    rand = uniform(0, log_range, shape=(num_sampled,), dtype='float64')
    # make sure sampled_classes are in the range of [0, range_max)
    sampled_classes = (rand.exp() - 1).astype('int64') % range_max

    true_classes = true_classes.astype('float64')
    expected_prob_true = ((true_classes + 2.0) / (true_classes + 1.0)).log() / log_range
    expected_count_true = expected_prob_true * num_sampled
    # cast sampled classes to fp64 to avoid interget division
    sampled_cls_fp64 = sampled_classes.astype('float64')
    expected_prob_sampled = ((sampled_cls_fp64 + 2.0) / (sampled_cls_fp64 + 1.0)).log() / log_range
    expected_count_sampled = expected_prob_sampled * num_sampled
    return sampled_classes, expected_count_true, expected_count_sampled

def _get_graph_inputs(subg):
    num_handles = ctypes.c_int(0)
    handles = ctypes.POINTER(SymbolHandle)()
    check_call(_LIB.MXSymbolGetInputSymbols(subg.handle, ctypes.byref(handles),
                                            ctypes.byref(num_handles)))

    syms = []
    for i in range(num_handles.value):
        s = Symbol(ctypes.cast(handles[i], SymbolHandle))
        syms.append(s)
    return syms

def _cut_subgraph(subg):
    num_handles = ctypes.c_int(0)
    handles = ctypes.POINTER(SymbolHandle)()
    check_call(_LIB.MXSymbolCutSubgraph(subg.handle, ctypes.byref(handles),
                                        ctypes.byref(num_handles)))

    syms = []
    for i in range(num_handles.value):
        s = Symbol(ctypes.cast(handles[i], SymbolHandle))
        syms.append(s)
    return syms

# This construct a subgraph for given output nodes.
# If an output node is one of the input nodes, we call identity to make sure
# that outputs nodes are different from input nodes.
def _construct_subgraph(sym_out, sym_states):
    sym_out = _as_list(sym_out)
    sym_states = _as_list(sym_states)
    all_outputs = []
    all_outputs.extend(sym_out)
    all_outputs.extend(sym_states)
    g = symbol.Group(all_outputs)

    flat_out = []
    all_input_names = g.list_inputs()
    output_names = [o.name for o in sym_out]
    for o in sym_out:
        if o.name in all_input_names:
            flat_out.append(symbol.op.identity(o))
        else:
            flat_out.append(o)

    for s in sym_states:
        if s.name in all_input_names or s.name in output_names:
            # There is a problem if the outputs are the same as the inputs
            # or the first output. By calling identity, we can make sure that
            # all symbols will refer to different NDArrays.
            flat_out.append(symbol.op.identity(s))
        else:
            flat_out.append(s)
    return symbol.Group(flat_out)

def foreach(body, data, init_states, name="foreach"):
    """Run a for loop with user-defined computation over Symbols on dimension 0.

    This operator simulates a for loop and body has the computation for an iteration
    of the for loop. It runs the computation in body on each slice from the input
    NDArrays.

    body takes two arguments as input and outputs a tuple of two elements,
    as illustrated below:

    out, states = body(data1, states)

    data1 can be either a symbol or a list of symbols. If data is a symbol,
    data1 is a symbol. Otherwise, data1 is a list of symbols and has the same
    size as data. states is a list of symbols and have the same size as init_states.
    Similarly, out can be either a symbol or a list of symbols, which are concatenated
    as the first output of foreach; states from the last execution of body
    are the second output of foreach.

    foreach can output only output data or states. If a user only wants states,
    the body function can return ([], states). Similarly, if a user only wants
    output data, the body function can return (out, []).

    The computation done by this operator is equivalent to the pseudo code below
    when the input data is NDArray:

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
    data: a symbol or a list of symbols.
        The input data.
    init_states: a symbol or a list of symbols.
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
    >>> data = mx.sym.var('data')
    >>> states = [mx.sym.var('state')]
    >>> outs, states = mx.sym.contrib.foreach(step, data, states)
    """

    def check_data(inputs, in_type, msg):
        is_NDArray_or_list = True
        if isinstance(inputs, list):
            for i in inputs:
                if not isinstance(i, in_type):
                    is_NDArray_or_list = False
                    break
        else:
            is_NDArray_or_list = isinstance(inputs, in_type)
        assert is_NDArray_or_list, msg

    check_data(data, symbol.Symbol, "data should be a symbol or a list of symbols")
    check_data(init_states, symbol.Symbol, "init_states should be a symbol or a list of symbols")
    not_state_list = isinstance(init_states, symbol.Symbol)

    # If the input python function references to the symbols outside
    # the python function, we need to prune the computation graph constructed from
    # the function. One way of doing it is to mark the nodes in the computation graph
    # with AttrScope and prune the nodes without the special attribute.
    with AttrScope(__subgraph_name__=name):
        if isinstance(data, list):
            in_eles = [symbol.var(sym.name) for sym in data]
        else:
            in_eles = symbol.var(data.name)
        if isinstance(init_states, list):
            states = [symbol.var(s.name) for s in init_states]
        else:
            states = symbol.var(init_states.name)
        sym_out, sym_states = body(in_eles, states)

        check_data(sym_out, symbol.Symbol,
                   "the output should be an NDArray or a list of NDArrays")
        check_data(sym_states, symbol.Symbol,
                   "the output states should be an NDArray or a list of NDArrays")
        if isinstance(sym_states, list):
            assert isinstance(init_states, list) and len(sym_states) == len(init_states), \
                    "the number of output states (%d) should be the same as input states (%d)" \
                    % (len(sym_states), len(init_states))
        num_out_data = len(sym_out)
        num_states = len(sym_states)
        num_outputs = num_out_data + num_states
        g = _construct_subgraph(sym_out, sym_states)

    input_syms = _get_graph_inputs(g)
    cut_syms = _cut_subgraph(g)
    input_syms = _get_graph_inputs(g)

    # Here we need to find out how the input symbols are ordered as well as
    # where the loop states are located in the list of inputs.

    # This dict contains the symbols of the subgraph.
    input_syms = {sym.name:sym for sym in input_syms}
    gin_names = input_syms.keys()
    # This array contains the symbols for the inputs of foreach.
    # They are ordered according to the inputs of the subgraph.
    init_states = _as_list(init_states)
    state_names = [sym.name for sym in init_states]
    data_syms = _as_list(data)
    data_names = [sym.name for sym in data_syms]
    cut_var_map = {sym.list_outputs()[0]:sym for sym in cut_syms}
    cut_var_names = cut_var_map.keys()

    subg_input_names = g.list_inputs()
    # ordered_ins contains input symbols in the following order:
    # data_syms, state_syms, followed by cut_vars and vars in the closure.
    ordered_ins = data_syms
    # this defines the location of data_syms in the list of subgraph inputs
    in_data_locs = []
    for dname in data_names:
        # Some data may not be used.
        if dname in subg_input_names:
            in_data_locs.append(subg_input_names.index(dname))
        else:
            raise AssertionError("the data arrays have to be used in the loop body")

    ordered_ins.extend(init_states)
    # this defines the location of state_syms in the list of subgraph inputs.
    in_state_locs = []
    for sname in state_names:
        # Some state may not be used.
        if sname in subg_input_names:
            in_state_locs.append(subg_input_names.index(sname))
        else:
            raise AssertionError("the state arrays have to be used in the loop body")

    remain_locs = []
    for in_name in subg_input_names:
        assert in_name in gin_names, "The input variable %s can't be found in graph inputs: %s" \
                % (in_name, str(gin_names))
        if in_name in cut_var_names:
            ordered_ins.append(cut_var_map[in_name])
            remain_locs.append(subg_input_names.index(in_name))
        elif in_name not in data_names and in_name not in state_names:
            # The remaining inputs are the variable nodes created inside the UDF.
            # The subgraph can't have nodes shared with the main graph. As such,
            # we need to make a copy of these variable nodes.
            assert in_name in gin_names
            ordered_ins.append(copy.deepcopy(input_syms[in_name]))
            remain_locs.append(subg_input_names.index(in_name))

    ret = symbol._internal._foreach(g, *ordered_ins, num_outputs=num_outputs,
                                    num_out_data=num_out_data, in_state_locs=in_state_locs,
                                    in_data_locs=in_data_locs, remain_locs=remain_locs)
    if num_outputs - num_states > 1:
        outs = []
        for i in range(num_outputs - num_states):
            outs.append(ret[i])
    elif num_outputs - num_states == 1:
        outs = ret[0]
    else:
        outs = []
    states = []
    for i in range(num_states):
        states.append(ret[num_outputs - num_states + i])

    if not_state_list:
        # If there is only one input state, there should be only one output state.
        assert len(states) == 1
        states = states[0]

    return (outs, states)
