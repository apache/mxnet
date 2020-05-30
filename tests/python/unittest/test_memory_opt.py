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
    exec = z.simple_bind(mx.cpu(), 'write', x=(num_hidden,), y=(num_hidden,))
    exec_debug_str = exec.debug_str().split('\n')
    op_checklist = 0
    for i, line in enumerate(exec_debug_str):
        if "Op:elemwise_add" in line:
            op_checklist += 1
            assert exec_debug_str[i + 5] == "\t__mirror_stage__=0"
        if "Op:Activation" in line:
            op_checklist += 1
            assert exec_debug_str[i + 4] == "\t__mirror_stage__=0"
    assert op_checklist == 2, \
           "Not all operator nodes have been verified on the mirror stage"


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
    exec = z.simple_bind(mx.cpu(), 'write', **in_arg_shapes)
    exec_debug_str = exec.debug_str().split('\n')
    op_checklist = 0
    for i, line in enumerate(exec_debug_str):
        for t in range(num_steps):
            if line == "Op:broadcast_add, Name=broadcast_add%d"%t:
                op_checklist += 1
                assert exec_debug_str[i + 5] == "\t__mirror_stage__=1"
            if line == "Op:Activation, Name=activation%d"%t:
                op_checklist += 1
                assert exec_debug_str[i + 4] == "\t__mirror_stage__=1"
    assert op_checklist == 2 * num_steps, \
           "Not all operator nodes have been verified on the mirror stage"


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
    exec = z.simple_bind(mx.cpu(), 'write', x=(num_hidden,))
    exec_debug_str = exec.debug_str().split('\n')
    op_checklist = 0
    for i, line in enumerate(exec_debug_str):
        if line == "Op:Activation, Name=y":
            op_checklist += 1
            assert exec_debug_str[i + 4] == "\t__mirror_stage__=0"
        if line == "Op:Activation, Name=z":
            op_checklist += 1
            assert exec_debug_str[i + 4] == "\t__mirror_stage__=1"
        if "Op:FullyConnected" in line:
            op_checklist += 1
            assert exec_debug_str[i + 6] == "\t__mirror_stage__=0"
        if "Op:_backward_FullyConnected" in line:
            op_checklist += 1
            assert exec_debug_str[i + 3] == "\targ[1]=z_mirror(0)"
    assert op_checklist == 4, \
           "Not all operator nodes have been verified on the mirror stage"


def grep_exec_memory_consumption(exec):
    # Grep the memory consumption (in MB) from the executor debug string.
    #
    # It is important to note that, due to various reasons, the memory
    # consumption reported by the executor debug string might be very different
    # when compared with the real numbers reported by nvidia-smi. These reasons
    # include:
    #   - Allocations by the CUDA Library (e.g., cuDNN, cuBLAS)
    #   - Fragmentation (of the MXNet Memory Allocator and cudaMalloc)
    exec_debug_str = exec.debug_str().split('\n')

    import re  # We will be using regular expressions for grepping the model
               # memory consumption.
    alloc_line_pattern = re.compile("Total \d+ MB allocated")
    for line in exec_debug_str:
        if alloc_line_pattern.match(line) is not None:
            return int(line.split()[1])
    assert False, "Unable to gerp the memory consumption numbers from the executor " \
                  "debug string: %s" % exec_debug_str


@memory_opt_env_check
def test_resnet152():
    # Verify the memory allocation behavior on ResNet-152, the state-of-the-art
    # model used for image classification.

    # Import the network, similar to what we did in
    # ${MXNET_ROOT_DIR}/example/image-classification/train_imagenet.py
    from importlib import import_module
    sys.path.append(os.path.join(os.path.dirname(__file__),
                    '..', '..', '..', 'example', 'image-classification'))
    resnet_mod = import_module('symbols.resnet')
    resnet_152 = resnet_mod.get_symbol(num_classes=1000,
                                       num_layers=152,
                                       image_shape='3,224,224')
    # We do the binding twice, one with the memory optimizations and one without.
    # It is expected that the memory consumption of the former should be roughly
    # half of that of the latter.
    memory_opt_exec = resnet_152.simple_bind(mx.cpu(), 'write',
                                             data=(32, 3, 224, 224))
    os.environ["MXNET_MEMORY_OPT"] = '0'
    no_opt_exec = resnet_152.simple_bind(mx.cpu(), 'write', data=(32, 3, 224, 224))
    os.environ["MXNET_MEMORY_OPT"] = '1'
    memory_opt_alloc = grep_exec_memory_consumption(memory_opt_exec)
    no_opt_alloc = grep_exec_memory_consumption(no_opt_exec)
    assert memory_opt_alloc / no_opt_alloc < 0.6, \
           "The ratio between the memory consumption with the memory optimizations " \
           "enabled and disabled (%d vs. %d MB) is expected to be smaller than 0.6"  \
           % (memory_opt_alloc, no_opt_alloc)


if __name__ == "__main__":
    import nose
    nose.runmodule()
