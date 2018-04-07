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

def foreach(func, input, init_states, back_prop=False):
    in_ele = mx.sym.var("in")
    gin_names = ["in"]
    states = []
    i = 0
    assert isinstance(init_states, list), "init_states should be a list"
    for s in init_states:
        states.append(mx.sym.var("state" + str(i)))
        gin_names.append("state" + str(i))
        i = i + 1
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

    # The input function can't have free variables right now.
    for i in g.list_inputs():
        assert i in gin_names, "The input function can't contain free variables"

    return mx.sym._internal._foreach(g, input, *init_states)
