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

import ctypes

from .. import symbol
from ..base import _LIB, c_str, c_array, check_call
from ..base import SymbolHandle, NDArrayHandle
from ..attribute import AttrScope

def _get_graph_inputs(subg, name, prefix):
    num_handles = ctypes.c_int(1000)
    handles = c_array(SymbolHandle, [SymbolHandle(0) for i in range(1000)])
    check_call(_LIB.MXSymbolGetInputSymbols(subg.handle, handles,
        ctypes.byref(num_handles)))

    syms = []
    for i in range(num_handles.value):
        s = symbol.Symbol(handles[i])
        syms.append(s)
    return syms

def foreach(func, input, init_states, back_prop=False, name="foreach"):
    assert isinstance(init_states, list), "init_states should be a list"
    states = []
    with AttrScope(subgraph_name=name):
        in_ele = symbol.var("in")
        for s in init_states:
            states.append(symbol.var(s.name))

        sym_out = func(in_ele, states)
        # The function should return a tuple. The first element goes to
        # the output of the function. The second element is a list.
        assert isinstance(sym_out, tuple), "func should return a tuple (out, states)"
        assert isinstance(sym_out[1], list), \
                "the second element in the returned tuple should be a list"
        assert len(sym_out[1]) == len(init_states), \
                "the number of output states (%d) should be the same as input states (%d)" \
                % (len(sym_out[1]), len(init_states))

        if (isinstance(sym_out[0], list)):
            flat_out = sym_out[0]
        else:
            flat_out = [sym_out[0]]
        for s in sym_out[1]:
            # There is a problem if the outputs are the same as the inputs
            # or the first output.
            # TODO this is a temp fix.
            flat_out.append(symbol.identity(s))
    g = symbol.Group(flat_out)
    input_syms = _get_graph_inputs(g, name, "ro_var")

    if (isinstance(input, list)):
        num_inputs = len(input)
    else:
        num_inputs = 1

    # Here we need to find out how the input symbols are ordered as well as
    # where the loop states are located in the list of inputs.

    # This dict contains the symbols of the subgraph.
    input_syms = {sym.name:sym for sym in input_syms}
    gin_names = input_syms.keys()
    # This array contains the symbols for the inputs of foreach.
    ordered_ins = []
    states_map = {sym.name:sym for sym in init_states}
    state_names = states_map.keys()
    in_state_locs = [-1] * len(init_states)
    for in_name in g.list_inputs():
        assert in_name in gin_names, "The input variable %s can't be found in graph inputs: %s" \
                % (in_name, str(gin_names))
        if (in_name in state_names):
            ordered_ins.append(states_map[in_name])
        elif (in_name != "in"):
            ordered_ins.append(input_syms[in_name])

        for i in range(len(init_states)):
            if (init_states[i].name == in_name):
                in_state_locs[i] = len(ordered_ins) - 1 + num_inputs

    num_outputs = len(flat_out)
    num_states = len(state_names)
    ret = symbol._internal._foreach(g, input, *ordered_ins, num_outputs=num_outputs,
                                    in_state_locs=in_state_locs)
    if (num_outputs - num_states > 1):
        outs = []
        for i in range(num_outputs - num_states):
            outs.append(ret[i])
    else:
        outs = ret[0]
    states = []
    for i in range(num_states):
        states.append(ret[num_outputs - num_states + i])

    return (outs, states)
