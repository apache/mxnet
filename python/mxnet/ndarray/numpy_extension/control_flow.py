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

from . import _api_internal
from .._internal import NDArrayBase
from ...util import set_module
from ...numpy import ndarray as np_ndarray
from ...symbol import Symbol
from ...base import _as_list
from ... import symbol, _deferred_compute as dc, autograd as ag
from ...attribute import AttrScope, current as current_attribute


__all__ = ["foreach", "while_loop", "cond"]


def _flatten(args, inout_str):
    """Parse the arguments into a flattened list + an additional format array.
    The format array stores the structure of the original arguments to help reconstruct the inputs.

    Parameters
    ----------
    args : NDArray, Symbol, or (nested) list of Symbol or NDArray
        We allow None inside the args.
    inout_str : str
        The name of the HybridBlock

    Returns
    -------
    flat : list of Symbol or NDArray
        The flatten version of the input args.
    fmts : (nested) list of ints
        Stores the format information of the original structured args.
    """
    if isinstance(args, np_ndarray):
        return [args], int(0)
    if isinstance(args, Symbol):
        length = len(args.list_outputs())
        length = length if length > 1 else 0
        return [args], int(length)
    if args is None:
        return [None], int(-1)

    if not isinstance(args, (list, tuple)):
        raise ValueError("When hybridized, the input of HybridBlock {}"
                         " must be (nested) list of Symbol"
                         " or NDArray, "
                         "but got {} of type {}".format(inout_str, str(args), str(type(args))))
    flat = []
    fmts = []
    for i in args:
        arg, fmt = _flatten(i, inout_str)
        flat.extend(arg)
        fmts.append(fmt)
    return flat, fmts


def _regroup(args, fmt):
    """Reconstruct the structured arguments based on the flattened version.

    Parameters
    ----------
    args : NDArray, Symbol, or (nested) list of Symbol or NDArray
        We allow None inside the args.
    fmt : (nested) list of ints
        Stores the format information of the original structured args.

    Returns
    -------
    ret : NDArray, Symbol, or (nested) list of Symbol or NDArray

    """
    def _merger(args, fmt):
        """Recursive call to merge the arguments"""
        if isinstance(fmt, int):
            if fmt < -1:
                raise ValueError("Unsupported encoded format {}.".format(fmt))
            if fmt == 0:
                return args[0], args[1:]
            if fmt == -1:
                if args[0] is not None:
                    raise ValueError('We do not support passing types that are not None'
                                     ' when the initial HybridBlock has received NoneType and'
                                     ' has been hybridized.'
                                     ' Received arg = {}, fmt = {}.'.format(args[0], fmt))
                return None, args[1:]
            else:
                return args[:fmt], args[fmt:]

        if not isinstance(args, (list, tuple)):
            raise ValueError("When hybridized, the output of HybridBlock must be (nested)"
                             " list of Symbol or NDArray, "
                             "but got {} of type {}".format(args, type(args)))
        ret = []
        for i in fmt:
            res, args = _merger(args, i)
            ret.append(res)
        return ret, args
    return _merger(args, fmt)[0]

def _get_unique_subgraph_name(subgraph_name):
    attrs = current_attribute()._attr
    if attrs.get("__subgraph_name__", "") != "":
        subgraph_name = "".join([attrs["__subgraph_name__"], "$", subgraph_name])
    AttrScope._subgraph_names[subgraph_name] += 1
    subgraph_name = subgraph_name + str(AttrScope._subgraph_names[subgraph_name] - 1)
    return subgraph_name

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
    output_names = {o.name for o in sym_out}
    for o in sym_out:
        if o.name in all_input_names:
            flat_out.append(symbol.op.identity(o))
        else:
            flat_out.append(o)

    for s in sym_states:
        if s.name in all_input_names or s.name in output_names:
            flat_out.append(symbol.op.identity(s))
        else:
            flat_out.append(s)
    return symbol.Group(flat_out)

@set_module('mxnet.ndarray.numpy_extension')
def foreach(body, data, init_states, name="foreach"):
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

    flatten_data, data_fmt = _flatten(data, "foreach input")
    check_input(flatten_data, np_ndarray,
                "data should be an mxnet.numpy.ndarray or a nested list of mxnet.numpy.ndarray")
    flatten_state, state_fmt = _flatten(init_states, "foreach states")
    check_input(flatten_state, np_ndarray,
                "init_states should be an mxnet.numpy.ndarray or a nested list of mxnet.numpy.ndarray")

    real_data = [ele[0].copy().detach() if ele is not None else None for ele in flatten_data]
    real_state = [ele.copy().detach() if ele is not None else None for ele in flatten_state]

    # If the input python function references to the symbols outside
    # the python function, we need to prune the computation graph constructed from
    # the function. One way of doing it is to mark the nodes in the computation graph
    # with AttrScope and prune the nodes without the special attribute.
    name = _get_unique_subgraph_name(name)
    with AttrScope(__subgraph_name__=name):
        data_names = ['data_subgraph{}'.format(i) for i, ele in enumerate(real_data)]
        state_names = ['state_subgraph{}'.format(i) for i, ele in enumerate(real_state)]
        symbol_data = [
            symbol.var(name).as_np_ndarray()
            for arg, name in zip(real_data, data_names)
        ]
        symbol_state = [
            symbol.var(name).as_np_ndarray()
            for arg, name in zip(real_state, state_names)
        ]
        dc.set_variable(real_data, symbol_data)
        dc.set_variable(real_state, symbol_state)
        in_eles = _regroup(real_data, data_fmt)
        in_states = _regroup(real_state, state_fmt)
        if dc.is_deferred_compute():
            out, states = body(in_eles, in_states)
        else:
            with ag.pause(), dc.context():
                out, states = body(in_eles, in_states)

        flatten_out, out_fmt = _flatten(out, "foreach output")
        flatten_out_state, state_fmt = _flatten(states, "foreach loop_vars")

        num_out_data = len(flatten_out)
        num_states = len(flatten_out_state)
        num_outputs = num_out_data + num_states
        sym_out = [dc.get_symbol(out_data) for out_data in flatten_out]
        sym_states = [dc.get_symbol(out_state) for out_state in flatten_out_state]
        dc.clear(flatten_out)
        dc.clear(flatten_out_state)
        g = _construct_subgraph(sym_out, sym_states)

    params_names = []
    params_data = []
    if hasattr(body, "collect_params"):
        for p in body.collect_params().values():
            params_names.append(p.var().name)
            params_data.append(p.data())

    subg_input_names = g.list_inputs()

    in_data, in_states, params = [], [], []
    in_data_locs, in_state_locs, remain_locs, in_state_index = [], [], [], []
    for i, sub_name in enumerate(subg_input_names):
        if sub_name in data_names:
            in_data_locs.append(i)
            idx = data_names.index(sub_name)
            in_data.append(flatten_data[idx])
        elif sub_name in state_names:
            in_state_locs.append(i)
            idx = state_names.index(sub_name)
            in_states.append(flatten_state[idx])
            in_state_index.append(idx)
        elif sub_name in params_names:
            remain_locs.append(i)
            idx = params_names.index(sub_name)
            params.append(params_data[idx])
        else:
            raise AssertionError("the data arrays have to be used in the loop body")

    ordered_ins = in_data + in_states + params

    ndoutput = _api_internal.foreach(g.handle, *ordered_ins, num_outputs, num_out_data, in_state_locs,
                                     in_data_locs, remain_locs, in_state_index)
    if isinstance(ndoutput, NDArrayBase):
        ret = ndoutput
    else:
        ret = list(ndoutput)
    outs = []
    for i in range(num_outputs - num_states):
        outs.append(ret[i])
    outs = _regroup(outs, out_fmt)
    states = []
    for i in range(num_states):
        states.append(ret[num_outputs - num_states + i])
    states = _regroup(states, state_fmt)

    return (outs, states)


#pylint: disable=W0621
@set_module('mxnet.ndarray.numpy_extension')
def while_loop(cond, func, loop_vars, max_iterations=None, name="while_loop"):
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
    def _to_python_scalar(inputs, type_, name):
        """Converts "inputs", possibly typed mxnet NDArray, a numpy ndarray, other python types,
        to the given type
        """
        if isinstance(inputs, np_ndarray):
            inputs = inputs.item()
        try:
            inputs = type_(inputs)
        except:
            raise ValueError(f"Cannot convert {name} to python {type_.__name__}")
        return inputs

    def _cond_wrapper(loop_vars):
        if dc.is_deferred_compute():
            result = cond(*loop_vars).astype("int")
        else:
            with ag.pause(), dc.context():
                result = cond(*loop_vars).astype("int")
        flatten_out, _ = _flatten(result, "while_loop output")
        out = dc.get_symbol(flatten_out)
        dc.clear(flatten_out)
        return [], [out], [], []

    def _func_wrapper(loop_vars):
        """This wrapper unifies
             "func: loop_vars -> new_loop_vars"
         and "func: loop_vars -> (step_output, new_loop_vars)"
        into "func: loop_vars -> (None or tuple of step_outputs, tuple of new_loop_vars)
        """
        if dc.is_deferred_compute():
            step_output, new_loop_vars = func(*loop_vars)
        else:
            with ag.pause(), dc.context():
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
        step_output_flatten, out_fmt = _flatten(step_output, "while output")
        new_loop_vars_flatten, var_fmt = _flatten(new_loop_vars, "while loop_vars")
        if isinstance(step_output, list):
            if len(step_output) == 0:
                step_out = []
            else:
                step_out = [dc.get_symbol(out) for out in step_output_flatten]
        else:
            step_output_flatten, out_fmt = _flatten(step_output, "while output")
            step_out = [dc.get_symbol(step_output_flatten)]
        if len(new_loop_vars) == 0:
            new_var = []
        else:
            new_var = [dc.get_symbol(var) for var in new_loop_vars_flatten]
        return step_out, new_var, out_fmt, var_fmt

    def _create_subgraph(graph_vars, graph_func, subgraph_name):
        subgraph_name = _get_unique_subgraph_name(subgraph_name)
        with AttrScope(__subgraph_name__=subgraph_name):
            # create new variables with the same name,
            # them feed them to the given func
            flatten_data, data_fmt = _flatten(graph_vars, "foreach input")
            real_data = [ele.copy().detach() if ele is not None else None for ele in flatten_data]
            data_names = ['data_subgraph{}'.format(i) for i, ele in enumerate(real_data)]
            symbol_data = [
                symbol.var(name).as_np_ndarray()
                for arg, name in zip(real_data, data_names)
            ]
            dc.set_variable(real_data, symbol_data)
            new_graph_vars = _regroup(real_data, data_fmt)
            outputs, final_state, out_fmt, var_fmt = graph_func(new_graph_vars)
            # first `num_out_data` elements belong to `outputs`
            # other elements belong to `final_state`
            num_out_data = len(outputs)
            num_outputs = len(outputs) + len(final_state)
            # group all outputs of graph_func
            graph = _construct_subgraph(outputs, final_state)
        return graph, num_out_data, num_outputs, out_fmt, var_fmt

    flatten_loop_vars, init_loop_var_fmt = _flatten(loop_vars, "while loop_vars")

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
            # some loop_vars are inputs to `graph`, some are not
            name_to_loop_vars = {'data_subgraph{}'.format(i): ele for i, ele in enumerate(flatten_loop_vars)}
            # also we collect the mapping from var's name to var's loc in loop_vars
            name_to_var_locs = {'data_subgraph{}'.format(i): i for i, ele in enumerate(flatten_loop_vars)}
            # collect arguments for each subgraph
            input_locs = []                         # results from the second step
            var_locs = [-1] * len(flatten_loop_vars)        # results from the third step
            subg_input_names = graph.list_inputs()
            for name in subg_input_names:
                assert name in name_to_loop_vars   # it should obviously hold
                array = name_to_loop_vars[name]
                # do 2), and 1) is implicitly done
                if id(array) in input_id_to_loc:
                    loc = input_id_to_loc[id(array)]
                else:
                    loc = len(input_id_to_loc)
                    inputs.append(array)
                    input_id_to_loc[id(array)] = loc
                input_locs.append(loc)
                # do 3)
                if name in name_to_var_locs:
                    var_locs[name_to_var_locs[name]] = len(input_locs) - 1
                    name_to_var_locs.pop(name, None)
            locs.append((input_locs, var_locs))
        return inputs, locs
    if max_iterations is None:
        raise ValueError("max_iterations should be specified")
    max_iterations = _to_python_scalar(max_iterations, int, "max_iteration")
    # It should be work as fine if loop_vars are empty I guess,
    # but it is semantically unnecessary to include this case.
    if isinstance(loop_vars, (list, tuple)):
        if len(loop_vars) == 0:
            raise ValueError("loop_vars should contain at least one element")
    else:
        assert isinstance(loop_vars, np_ndarray), ("loop_vars should be either mxnet.numpy.ndarray" \
            " or list/tuple of mxnet.numpy.ndarray")
        loop_vars = [loop_vars]
    # create graph for `cond'
    cond_g, num_out_data, num_outputs, _, _ = \
        _create_subgraph(loop_vars, _cond_wrapper, name + "_cond")
    assert num_out_data == 0
    assert num_outputs == 1
    # create graph for `func`
    func_g, num_out_data, num_outputs, out_fmt, _ = \
        _create_subgraph(loop_vars, _func_wrapper, name + "_func")
    # find symbols used in either cond_g or func_g
    input_vars, ((cond_input_locs, _), (func_input_locs, func_var_locs)) = \
        _union_inputs(cond_g, func_g)
    for i_th, loc in enumerate(func_var_locs, 1):
        if loc == -1:
            raise ValueError(f"The {i_th}-th loop_var doesn't involve into the computation")
    result = _api_internal.while_loop(
        cond_g.handle,
        func_g.handle,
        *input_vars,
        max_iterations,
        cond_input_locs,
        func_input_locs,
        func_var_locs,
        num_out_data,
        num_outputs
    )
    if isinstance(result, np_ndarray):
        ret = [result]
    else:
        ret = list(result)
    outputs = [ret[i] for i in range(num_out_data)]
    outputs = _regroup(outputs, out_fmt)
    final_loop_vars = [ret[i] for i in range(num_out_data, num_outputs)]
    final_loop_vars = _regroup(final_loop_vars, init_loop_var_fmt)
    return outputs, final_loop_vars


@set_module('mxnet.ndarray.numpy_extension')
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

    def _create_subgraph(graph_vars, graph_func, subgraph_name):
        subgraph_name = _get_unique_subgraph_name(subgraph_name)
        with AttrScope(__subgraph_name__=subgraph_name):
            # create new variables with the same name,
            # them feed them to the given func
            flatten_data, data_fmt = _flatten(graph_vars, "cond input")
            real_data = [ele.copy().detach() if ele is not None else None for ele in flatten_data]
            data_names = ['data_subgraph{}'.format(i) for i, ele in enumerate(real_data)]
            symbol_data = [
                symbol.var(name).as_np_ndarray()
                for arg, name in zip(real_data, data_names)
            ]
            dc.set_variable(real_data, symbol_data)
            new_graph_vars = _regroup(real_data, data_fmt)
            if dc.is_deferred_compute():
                outputs = graph_func(*new_graph_vars)
                if "pred" in subgraph_name:
                    outputs = outputs.astype("int")
            else:
                with ag.pause(), dc.context():
                    outputs = graph_func(*new_graph_vars)
                    if "pred" in subgraph_name:
                        outputs = outputs.astype("int")
            outputs, out_fmt = _flatten(outputs, "cond outputs")
            num_outputs = len(outputs)
            sym_out = [dc.get_symbol(out_data) for out_data in outputs]
            dc.clear(outputs)
            graph = _construct_subgraph(sym_out, [])
        return graph, num_outputs, out_fmt

    flatten_inputs, _ = _flatten(inputs, "while loop_vars")

    def _union_inputs(*graphs):
        # Given a list of graphs, each whose inputs are either from input_vars or other variables.
        # 1) calculate a list `inputs`, the union of their inputs.
        # 2) for each graph, determine in which indices their inputs reside in `inputs`
        # 3) for each variable in the input of `graph`, find which index it is
        inputs = []             # List[Symbol], result of 1)
        locs = []               # List[Tuple(List[Int], List[Int])], a list of tuples,
                                # where tuples are results of 2) and 3)
        input_id_to_loc = {}    # Dict[int, int], given id(sym), input_id_to_loc maps it
                                # to a `loc`, where inputs[loc] = sym
        for graph in graphs:
            # some input_vars are inputs to `graph`, some are not
            name_to_input_syms = {'data_subgraph{}'.format(i): ele for i, ele in enumerate(flatten_inputs)}
            # collect arguments for each subgraph
            input_locs = []                         # results from the second step
            for name in graph.list_inputs():
                assert name in name_to_input_syms   # it should obviously hold
                array = name_to_input_syms[name]
                # do 2), and 1) is implicitly done
                if id(array) in input_id_to_loc:
                    loc = input_id_to_loc[id(array)]
                else:
                    loc = len(input_id_to_loc)
                    inputs.append(array)
                    input_id_to_loc[id(array)] = loc
                input_locs.append(loc)
            locs.append(input_locs)
        return inputs, locs
    if isinstance(inputs, (list, tuple)):
        if len(inputs) == 0:
            raise ValueError("inputs should contain at least one element")
    else:
        assert isinstance(inputs, np_ndarray), ("inputs should be either mxnet.numpy.ndarray" \
            " or list/tuple of mxnet.numpy.ndarray")
        inputs = [inputs]
    # create graph for `cond_func'
    cond_g, cond_num_outputs, _ = _create_subgraph(inputs, pred, name + "_pred")
    if cond_num_outputs != 1:
        raise ValueError("pred should always be a single output")
    # create graph for `then`
    then_g, then_num_outputs, then_fmt = _create_subgraph(inputs, then_func, name + "_then")
    # create graph for `else`
    else_g, else_num_outputs, _ = _create_subgraph(inputs, else_func, name + "_else")
    if then_num_outputs != else_num_outputs:
        raise ValueError("Number of outputs differs between then-branch and else-branch")
    # find symbols used in either cond_g or func_g
    union_inputs, (cond_input_locs, then_input_locs, else_input_locs) = \
        _union_inputs(cond_g, then_g, else_g)
    result = _api_internal.cond(
        cond_g.handle,
        then_g.handle,
        else_g.handle,
        *union_inputs,
        cond_input_locs,
        then_input_locs,
        else_input_locs,
        then_num_outputs
    )
    if isinstance(result, np_ndarray):
        ret = [result]
    else:
        ret = list(result)
    outputs = [ret[i] for i in range(then_num_outputs)]
    outputs = _regroup(outputs, then_fmt)
    return outputs
