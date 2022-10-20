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

# pylint: skip-file
import mxnet as mx
import os
import sys
from common import with_environment
from mxnet.test_utils import environment

num_hidden = 4096

@with_environment('MXNET_MEMORY_OPT', '1')
def test_rnn_cell():
    # x →→→ + →→→ tanh ⇒⇒⇒
    #       ↑
    # y →→→→
    #
    # ⇒⇒⇒ : Backward Dependency
    # In this example, there is no benefit in mirroring the elementwise-add
    # operator and the tanh operator.
    x = mx.sym.Variable("x")
    x = mx.sym.FullyConnected(x, num_hidden=num_hidden)
    y = mx.sym.Variable("y")
    y = mx.sym.FullyConnected(y, num_hidden=num_hidden)
    tmp = mx.sym._internal._plus(x, y)
    z = mx.sym.Activation(tmp, act_type='tanh')
    exec = z._simple_bind(mx.cpu(), 'write', x=(num_hidden,), y=(num_hidden,))


@with_environment('MXNET_MEMORY_OPT', '1')
def test_mlp_attn():
    # x →→→ + →→→ tanh ⇒⇒⇒
    #       ↑ + →→→ tanh ⇒⇒⇒
    # y_1 →→  ↑ + →→→ tanh ⇒⇒⇒
    # y_2 →→→→  ↑ ⋱
    # y_3 →→→→→→    + →→→ tanh ⇒⇒⇒
    #               ↑
    # y_n →→→→→→→→→→
    x = mx.sym.Variable("x")
    tmp, z = [], []
    num_steps = 5
    in_arg_shapes = {'x': (num_steps, num_hidden,)}
    for i in range(num_steps):
        y = mx.sym.Variable(f"y_t{i}")
        tmp.append(mx.sym.broadcast_add(x, y, name=f"broadcast_add{i}"))
        z.append(mx.sym.Activation(tmp[-1], act_type='tanh',
                                   name=f"activation{i}"))
        in_arg_shapes[f"y_t{i}"] = (1, num_hidden,)
    z = mx.sym.Group(z)
    exec = z._simple_bind(mx.cpu(), 'write', **in_arg_shapes)


@with_environment('MXNET_MEMORY_OPT', '1')
def test_fc():
    # x →→→ tanh ⇒⇒⇒ tanh  ⇒⇒⇒ FC
    #            →→→ tanh_ →→→
    #                          ↓
    #                          FC'
    x = mx.sym.Variable("x")
    y = mx.sym.Activation(x, act_type='tanh', name='y')
    z = mx.sym.Activation(y, act_type='tanh', name='z')
    z = mx.sym.FullyConnected(z, num_hidden=num_hidden)
    exec = z._simple_bind(mx.cpu(), 'write', x=(num_hidden,))
