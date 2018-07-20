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

__all__ = ["rand_zipfian", "foreach", "while_loop"]

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

def while_loop(cond, func, loop_vars, max_iterations=None, name="while_loop"):
    """Run a while loop with user-defined computation and loop condition.

    This operator simulates a while loop which iterately does customized computation
    as long as the condition is satisfied.

    `loop_vars` is a list of Symbols on which the computation uses.

    `cond` is a user-defined function, used as the loop condition.
    It consumes `loop_vars`, and produces a scalar MXNet symbol,
    indicating the termination of the loop.
    The loop ends when `cond` returns false (zero).
    The `cond` is variadic, and its signature should be
    `cond(*loop_vars) => Symbol`.

    `func` is a user-defined function, used as the loop body.
    It also consumes `loop_vars`, and produces `step_output` and `new_loop_vars` at each step.
    In each step, `step_output` should contain the same number elements.
    Through all steps, the i-th element of `step_output` should have the same shape and dtype.
    Also, `new_loop_vars` should contain the same number of elements as `loop_vars`,
    and the corresponding element should have the same shape and dtype.
    The `func` is variadic, and its signature should be
    `func(*loop_vars) => (List[Symbol] step_output, List[Symbol] new_loop_vars)`.

    `max_iterations` is a scalar that defines the maximum number of iterations allowed.

    This function returns two lists.
    The first list has the length of `|step_output|`,
    in which the i-th element are all i-th elements of
    `step_output` from all steps, stacked along axis 0.
    The second list has the length of `|loop_vars|`,
    which represents final states of loop variables.

    .. warning::

       For now, the axis 0 of all Symbols in the first list are `max_iterations`,
       due to lack of dynamic shape inference.

    .. warning::

       Even if `cond` is never satisfied,
       while_loop returns a list of outputs with inferred dtype and shape.
       This is different from the Symbol version,
       where in this case `step_outputs` are assumed as an empty list.

    Parameters
    ----------
    cond: a Python function.
        The loop condition.
    func: a Python function.
        The loop body.
    loop_vars: list of Symbol.
        The initial values of the loop variables.
    max_iterations: a python int.
        Maximum number of iterations.

    Returns
    ------
    outputs: list of Symbols
        stacked output from each step
    states: list of Symbols
        final state

    Examples
    --------
    >>> cond = lambda i, s: i <= 5
    >>> func = lambda i, s: ([i + s], [i + 1, s + i])
    >>> loop_vars = (mx.sym.var('i'), mx.sym.var('s'))
    >>> outputs, states = mx.sym.contrib.while_loop(cond, func, loop_vars, max_iterations=10)
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
            raise ValueError("Cannot convert %s to python %s" % (name, type_.__name__))
        return inputs

    def _to_symbol_tuple(inputs, name):
        """Converts "inputs", possibly a single mxnet Symbol, a list of mxnet Symbol,
        a tuple of mxnet Symbol, into a tuple of Symbol
        """
        if isinstance(inputs, list):
            inputs = tuple(inputs)
        if isinstance(inputs, Symbol):
            inputs = (inputs, )
        if not isinstance(inputs, tuple):
            raise ValueError("%s must be a Symbol, or a tuple or list of Symbol" % (name, ))
        for item in inputs:
            if not isinstance(item, Symbol):
                raise ValueError("%s must be a Symbol, or a tuple or list of Symbol" % (name, ))
        return inputs

    def _cond_wrapper(loop_vars):
        result = cond(*loop_vars)
        if not isinstance(result, Symbol):
            raise ValueError("Return of cond must be a Symbol")
        return [], [result]

    def _func_wrapper(loop_vars):
        """This wrapper unifies
             "func: loop_vars -> new_loop_vars"
         and "func: loop_vars -> (step_output, new_loop_vars)"
        into "func: loop_vars -> (list of step_outputs, tuple of new_loop_vars)
        """
        step_output, new_loop_vars = func(*loop_vars)
        if step_output is None:
            step_output = []
        if new_loop_vars is None:
            new_loop_vars = []
        step_output = _to_symbol_tuple(step_output, "step_output")
        new_loop_vars = _to_symbol_tuple(new_loop_vars, "new_loop_vars")
        if len(loop_vars) != len(new_loop_vars):
            raise ValueError("The number of loop_vars should be consistent during the loop")
        return list(step_output), list(new_loop_vars)

    def _create_subgraph(graph_vars, graph_func, subgraph_name):
        with AttrScope(__subgraph_name__=subgraph_name):
            # create new variables with the same name,
            # them feed them to the given func
            new_graph_vars = [symbol.var(sym.name) for sym in graph_vars]
            outputs, final_state = graph_func(new_graph_vars)
            # first `num_out_data` elements belong to `outputs`
            # other elements belong to `final_state`
            num_out_data = len(outputs)
            num_outputs = len(outputs) + len(final_state)
            # nnvm cut-graph does not allow inputs and outputs overlap
            # so we calculate the name of inputs, and copy outputs once it overlaps with inputs
            all_input_names = symbol.Group(outputs + final_state).list_inputs()
            make_identity = lambda x: symbol.op.identity(x) if x.name in all_input_names else x
            # group all outputs of graph_func
            graph = symbol.Group(list(map(make_identity, outputs + final_state)))
        return graph, num_out_data, num_outputs

    def _union_inputs(*graphs):
        # Given a list of graphs, each whose inputs are either from loop_vars or other variables.
        # 1) calculate a list `inputs`, the union of their inputs.
        # 2) for each graph, determine in which indices their inputs reside in `inputs`
        # 3) for each variable in the input of `graph`, find which index it is
        inputs = []             # List[Symbol], result of 1)
        locs = []               # List[Tuple(List[Int], List[Int])], a list of tuples,
                                # where tuples are results of 2) and 3)
        input_id_to_loc = {}    # Dict[int, int], given id(sym), input_id_to_loc maps it
                                # to a `loc`, where inputs[loc] = sym
        for graph in graphs:
            # input_syms: all inputs to the `graph`
            name_to_input_syms = {sym.name: sym for sym in _get_graph_inputs(graph)}
            # some loop_vars are inputs to `graph`, some are not
            name_to_loop_vars = {sym.name: sym for sym in loop_vars}
            # other inputs to `graph` created by cut_graph
            name_to_cut_g_syms = {sym.list_outputs()[0]: sym for sym in _cut_subgraph(graph)}
            # also we collect the mapping from var's name to var's loc in loop_vars
            name_to_var_locs = {sym.name: i for i, sym in enumerate(loop_vars)}
            # collect arguments for each subgraph
            input_locs = []                         # results from the second step
            var_locs = [-1] * len(loop_vars)        # results from the third step
            for name in graph.list_inputs():
                assert name in name_to_input_syms   # it should obviously hold
                # name -> sym
                if name in name_to_loop_vars:
                    sym = name_to_loop_vars[name]
                elif name in name_to_cut_g_syms:
                    sym = name_to_cut_g_syms[name]
                else:
                    sym = copy.deepcopy(name_to_input_syms[name])
                # do 2), and 1) is implicitly done
                if id(sym) in input_id_to_loc:
                    loc = input_id_to_loc[id(sym)]
                else:
                    loc = len(input_id_to_loc)
                    inputs.append(sym)
                    input_id_to_loc[id(sym)] = loc
                input_locs.append(loc)
                # do 3)
                if name in name_to_var_locs:
                    var_locs[name_to_var_locs[name]] = len(input_locs) - 1
            locs.append((input_locs, var_locs))
        return inputs, locs
    if max_iterations is None:
        raise ValueError("max_iterations should be specified")
    max_iterations = _to_python_scalar(max_iterations, int, "max_iteration")
    loop_vars = _to_symbol_tuple(loop_vars, "loop_vars")
    # It should be work as fine if loop_vars are empty I guess,
    # but it is semantically unnecessary to include this case.
    if len(loop_vars) == 0:
        raise ValueError("loop_vars should contain at least one element")
    # create graph for `cond'
    cond_g, num_out_data, num_outputs = \
        _create_subgraph(loop_vars, _cond_wrapper, name + "_cond")
    assert num_out_data == 0
    assert num_outputs == 1
    # create graph for `func`
    func_g, num_out_data, num_outputs = \
        _create_subgraph(loop_vars, _func_wrapper, name + "_func")
    # find symbols used in either cond_g or func_g
    input_syms, ((cond_input_locs, _), (func_input_locs, func_var_locs)) = \
        _union_inputs(cond_g, func_g)
    for i_th, loc in enumerate(func_var_locs, 1):
        if loc == -1:
            raise ValueError("The %d-th loop_var doesn't involve into the computation" % i_th)
    result = symbol._internal._while_loop(
        # [cond, func_g, *input_syms]
        cond_g,
        func_g,
        *input_syms,
        max_iterations=max_iterations,
        cond_input_locs=cond_input_locs,
        func_input_locs=func_input_locs,
        func_var_locs=func_var_locs,
        num_out_data=num_out_data,
        num_outputs=num_outputs
    )
    outputs = [result[i] for i in range(num_out_data)]
    final_loop_vars = [result[i] for i in range(num_out_data, num_outputs)]
    return outputs, final_loop_vars
