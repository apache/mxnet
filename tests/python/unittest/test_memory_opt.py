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


num_hidden = 4096


def memory_opt_env_check(test_func):
    # This decorator checks for th
    def test_memory_opt_wrapper():
        # Whether the underlying OS is Windows or not. Windows does not support
        # setting environment variblae on the fly. In other words, statement
        #
        #     os.environ["MXNET_MEMORY_OPT"] = '1'
        #
        # will have NO effect because the C++ backend still sees
        # `os.environ["MXNET_MEMORY_OPT"]` as NULL pointer.
        #
        # \sa test_operator.py:test_norm
        is_windows = sys.platform.startswith('win')
        do_memory_opt = True
        if is_windows:
            if "MXNET_MEMORY_OPT" not in os.environ:
                do_memory_opt = False
            else:
                do_memory_opt = os.environ["MXNET_MEMORY_OPT"] == '1'
        else:
            os.environ["MXNET_MEMORY_OPT"] = '1'

        if do_memory_opt:
            test_func()
            os.environ["MXNET_MEMORY_OPT"] = '0'
    return test_memory_opt_wrapper


@memory_opt_env_check
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


@memory_opt_env_check
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
        y = mx.sym.Variable("y_t%d"%i)
        tmp.append(mx.sym.broadcast_add(x, y, name="broadcast_add%d"%i))
        z.append(mx.sym.Activation(tmp[-1], act_type='tanh',
                                   name="activation%d"%i))
        in_arg_shapes["y_t%d"%i] = (1, num_hidden,)
    z = mx.sym.Group(z)
    exec = z._simple_bind(mx.cpu(), 'write', **in_arg_shapes)


@memory_opt_env_check
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


if __name__ == "__main__":
    import nose
    nose.runmodule()
