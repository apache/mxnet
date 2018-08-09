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
from common import *


def detect_cycle_from(sym, visited, stack):
    visited.add(sym.handle.value)
    stack.add(sym.handle.value)
    for s in sym.get_children():
        if s.handle.value not in visited:
            if detect_cycle_from(sym, visited, stack):
                return True
        elif s.handle.value in stack:
            return True
        stack.remove(sym.handle.value)
    return False


def has_no_cycle(sym):
    visited = set()
    stack = set()
    all_nodes = sym.get_internals()
    for s in all_nodes:
        if s.handle.value in visited:
            if detect_cycle_from(s, visited, stack):
                return False
    return True


def test_simple_cycle():
    inp = mx.sym.Variable('input', shape=[1,10])
    A = mx.sym.FullyConnected(data=inp, num_hidden=10, no_bias=False, name='A')
    B = mx.sym.FullyConnected(data=A, num_hidden=10, no_bias=False, name='B')
    D = mx.sym.sin(data=A, name='D')
    C = mx.sym.elemwise_add(lhs=B, rhs=D, name='C')
    arg_params = {
                'I_weight': mx.nd.zeros([10,10]),
                'I_bias': mx.nd.zeros([10]),
                'A_weight': mx.nd.zeros([10,10]),
                'A_bias': mx.nd.zeros([10]),
                'B_weight': mx.nd.zeros([10,10]),
                'B_bias': mx.nd.zeros([10]),
               }

    executor = C.simple_bind(ctx=mx.gpu(0), data=(1,10), softmax_label=(1,),
                           shared_buffer=arg_params, grad_req='null', force_rebind=True)
    optimized_graph = mx.contrib.tensorrt.get_optimized_symbol(executor)
    assert has_no_cycle(optimized_graph), "The graph optimized by TRT contains a cycle"


if __name__ == '__main__':
    import nose
    nose.runmodule()
