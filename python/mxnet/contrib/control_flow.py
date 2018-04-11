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

import mxnet as mx

def foreach(func, input, init_states, back_prop=False, name="foreach"):
    in_ele = mx.sym.var("in")
    gin_names = ["in"]
    states = []
    i = 0
    assert isinstance(init_states, list), "init_states should be a list"
    for s in init_states:
        states.append(mx.sym.var(s.name))
        gin_names.append(s.name)
        i = i + 1
    with mx.AttrScope(subgraph_name=name):
        sym_out = func(in_ele, states)
        # The function should return a tuple. The first element goes to
        # the output of the function. The second element is a list.
        assert isinstance(sym_out, tuple), "func should return a tuple (out, states)"
        assert isinstance(sym_out[1], list), \
                "the second element in the returned tuple should be a list"

        flat_out = [sym_out[0]]
        for s in sym_out[1]:
            # There is a problem if the outputs are the same as the inputs
            # or the first output.
            # TODO this is a temp fix.
            flat_out.append(mx.sym.identity(s))
    g = mx.sym.Group(flat_out)

    # Find free variables in the python that are symbols.
    freevars = dict(zip(func.func_code.co_freevars,
        (c.cell_contents for c in func.func_closure)))
    sym_freevars = []
    for name in freevars:
        val = freevars[name]
        if isinstance(val, mx.sym.Symbol):
            # We need to save the original symbol first.
            sym_freevars.append(val)
            gin_names.append(name)

    if (isinstance(input, list)):
        num_inputs = len(input)
    else:
        num_inputs = 1

    # Here we need to find out how the input symbols are ordered as well as
    # where the loop states are located in the list of inputs.
    ins = init_states + sym_freevars
    ins = {sym.name:sym for sym in ins}
    ordered_ins = []
    in_state_locs = [-1] * len(init_states)
    for in_name in g.list_inputs():
        assert in_name in gin_names, "The input graph contains variables we can't find"
        if in_name in ins:
            ordered_ins.append(ins[in_name])
            for i in range(len(init_states)):
                if (init_states[i].name == in_name):
                    in_state_locs[i] = len(ordered_ins) - 1 + num_inputs

    return mx.sym._internal._foreach(g, input, *ordered_ins, num_outputs=len(flat_out),
                                     in_state_locs=in_state_locs)
