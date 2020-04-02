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

import os
import ctypes
import mxnet as mx
from mxnet.base import SymbolHandle, check_call, _LIB, mx_uint, c_str_array, c_str, mx_real_t
from mxnet.symbol import Symbol
import numpy as np
from mxnet.test_utils import assert_almost_equal
from mxnet import gluon
from mxnet.gluon import nn
from mxnet import nd

def network_structure_1():
    data1 = mx.sym.var('data1', shape=(2, 3, 10, 10))
    data2 = mx.sym.var('data2')
    conv1 = mx.sym.Convolution(data=data1, weight=data2, no_bias=True, kernel=(2, 2), num_filter=1)
    conv2 = mx.sym.Convolution(data=data2, no_bias=True, kernel=(1, 1), num_filter=1)
    out = mx.sym.Group([conv1, conv2])
    return (out, ['data1'], [(2, 3, 10, 10)])

def network_structure_2():
    # this tests whether the partitioning algorithm can deal with cycles
    data = mx.sym.var('data', shape=(2, 3, 10, 10))
    ret = mx.sym.exp(data)
    ret1 = mx.sym.cos(ret)
    ret2 = mx.sym.sin(ret)
    ret = ret1 + ret2
    return (ret, ['data'], [(2, 3, 10, 10)]) 

def network_structure_3():
    # this tests whether the partitioned sym can distinguish in_args and aux_states
    data = mx.sym.var('data', shape=(2, 3, 10, 10))
    ret = mx.sym.exp(data)
    ret1 = mx.sym.cos(ret)
    ret2 = mx.sym.sin(ret)
    ret = ret1 + ret2
    ret = mx.sym.BatchNorm(ret)
    ret = mx.sym.BatchNorm(ret)
    # Return the same and shape of 'data' and auxiliary states
    return  (ret, ['data'] + ret.list_auxiliary_states(), [(2, 3, 10, 10), (3,), (3,), (3,), (3,)])

def network_structure_4():
    # the last op has multiple duplicate outputs
    data = mx.sym.var('data', shape=(2, 3, 10, 10))
    ret = mx.sym.exp(data)
    ret = mx.sym.Group([ret, ret, ret])
    return (ret, ['data'], [(2, 3, 10, 10)])

def network_structure_5():
    # the subgraph has two duplicate input entries
    data = mx.sym.var('data', shape=(2, 3, 10, 10))
    ret = data + data
    return (ret, ['data'], [(2, 3, 10, 10)])

def network_structure_6():
    data1 = mx.sym.Variable('data1', shape=(3, 3, 10, 10), dtype=np.float32)
    data2 = mx.sym.Variable('data2', shape=(1, 0, 2, 2))
    data3 = mx.sym.sin(data2)
    conv = mx.sym.Convolution(data=data1, weight=data3, kernel=(2, 2), num_filter=1)
    return (conv, ['data1'], [(3, 3, 10, 10)])
        
def network_structure_7():
    # in this graph, the subgraph node and the other two external nodes form a cycle
    data = mx.sym.Variable('data', shape=(1,))
    ret1 = mx.sym.sin(data)
    ret2 = mx.sym.cos(ret1)
    for _ in range(5):
        ret2 = mx.sym.cos(ret2)
    ret = ret1 + ret2
    return (ret, ['data'], [(1,)])

def get_graphs(): 
    return [
            (network_structure_1(), ['Convolution']),
            (network_structure_2(), ['exp', 'sin', '_Plus', 'elemwise_add', '_plus']),
            (network_structure_2(), ['exp', 'cos', '_Plus', 'elemwise_add', '_plus']),
            (network_structure_3(), ['exp', 'sin', '_Plus', 'elemwise_add', '_plus']),
            (network_structure_3(), ['exp', 'cos', '_Plus', 'elemwise_add', '_plus']),
            (network_structure_3(), ['exp', 'sin', '_Plus', 'elemwise_add', '_plus', 'BatchNorm']),
            (network_structure_3(), ['exp', 'cos', '_Plus', 'elemwise_add', '_plus', 'BatchNorm']),
            (network_structure_3(), ['exp', 'BatchNorm']),
            (network_structure_3(), ['BatchNorm']),
            (network_structure_4(), ['exp']),
            (network_structure_5(), ['_plus', '_Plus', 'elemwise_add']),
            (network_structure_6(), []),
            (network_structure_6(), [mx.sym.sin.__name__]),
            (network_structure_6(), [mx.sym.Convolution.__name__]),
            (network_structure_6(), [mx.sym.sin.__name__, mx.sym.Convolution.__name__]),
            (network_structure_7(), ['sin', 'elemwise_add', '_plus', '_Plus'])
            ]

def check_subgraph_exe1(sym, subgraph_backend, op_names):
    """Use the partitioned sym to simple_bind an executor and compare the outputs
    with those of the original executor"""
    out = SymbolHandle()
    check_call(_LIB.MXBuildSubgraphByOpNames(sym.handle, c_str(subgraph_backend), mx_uint(len(op_names)),
                                              c_str_array(op_names), ctypes.byref(out)))

    partitioned_sym = Symbol(out)
    assert partitioned_sym.list_inputs() == sym.list_inputs()
    assert partitioned_sym.list_arguments() == sym.list_arguments()
    assert partitioned_sym.list_auxiliary_states() == sym.list_auxiliary_states()
    exe = sym.simple_bind(ctx=mx.current_context(), grad_req='null')
    partitioned_exe = partitioned_sym.simple_bind(ctx=mx.current_context(), grad_req='null')
    input_names = sym.list_inputs()
    for name in input_names:
        if name in exe.arg_dict:
            exe.arg_dict[name][:] = mx.nd.random.uniform(shape=exe.arg_dict[name].shape)
            partitioned_exe.arg_dict[name][:] = exe.arg_dict[name]
        else:
            assert name in exe.aux_dict
            exe.aux_dict[name][:] = mx.nd.random.uniform(shape=exe.aux_dict[name].shape)
            partitioned_exe.aux_dict[name][:] = exe.aux_dict[name]
    exe.forward()
    partitioned_exe.forward()
    assert len(exe.outputs) == len(partitioned_exe.outputs)
    for i in range(len(exe.outputs)):
        assert_almost_equal((exe.outputs[i] - partitioned_exe.outputs[i]).abs().sum().asnumpy(),
                            np.zeros(shape=(1,)))

def check_subgraph_exe2(sym, subgraph_backend, op_names):
    """Use env var MXNET_SUBGRAPH_BACKEND=default to trigger graph partitioning in simple_bind
    and compare results of the partitioned sym and the original sym."""
    def get_executor(sym, subgraph_backend=None, op_names=None, original_exec=None):
        if subgraph_backend is not None:
            os.environ['MXNET_SUBGRAPH_BACKEND'] = subgraph_backend
            check_call(_LIB.MXSetSubgraphPropertyOpNames(c_str(subgraph_backend), mx_uint(len(op_names)),
                                                         c_str_array(op_names)))
        exe = sym.simple_bind(ctx=mx.current_context(), grad_req='null')
        input_names = sym.list_inputs()
        for name in input_names:
            if name in exe.arg_dict:
                exe.arg_dict[name][:] = mx.nd.random.uniform(shape=exe.arg_dict[name].shape)\
                    if original_exec is None else original_exec.arg_dict[name]
            else:
                assert name in exe.aux_dict
                exe.aux_dict[name][:] = mx.nd.random.uniform(shape=exe.aux_dict[name].shape)\
                    if original_exec is None else original_exec.aux_dict[name]
        exe.forward()
        if subgraph_backend is not None:
            check_call(_LIB.MXRemoveSubgraphPropertyOpNames(c_str(subgraph_backend)))
            del os.environ['MXNET_SUBGRAPH_BACKEND']
        return exe

    original_exec = get_executor(sym)
    partitioned_exec = get_executor(sym, subgraph_backend, op_names, original_exec)
    outputs1 = original_exec.outputs
    outputs2 = partitioned_exec.outputs
    assert len(outputs1) == len(outputs2)
    for i in range(len(outputs1)):
        assert_almost_equal((outputs1[i] - outputs2[i]).abs().sum().asnumpy(), np.zeros(shape=(1,)))

def check_subgraph_exe3(sym, subgraph_backend, op_names):
    """Use the partitioned sym to bind an executor and compare the outputs
    with those of the original executor"""
    out = SymbolHandle()
    check_call(_LIB.MXBuildSubgraphByOpNames(sym.handle, c_str(subgraph_backend), mx_uint(len(op_names)),
                                              c_str_array(op_names), ctypes.byref(out)))

    partitioned_sym = Symbol(out)
    input_names = sym.list_inputs()
    arg_names = sym.list_arguments()
    aux_names = sym.list_auxiliary_states()
    assert partitioned_sym.list_inputs() == input_names
    assert partitioned_sym.list_arguments() == arg_names
    assert partitioned_sym.list_auxiliary_states() == aux_names
    arg_shapes, _, aux_shapes = sym.infer_shape()
    arg_array = [mx.nd.random.uniform(shape=shape) for shape in arg_shapes]
    aux_array = [mx.nd.random.uniform(shape=shape) for shape in aux_shapes]
    exe = sym.bind(ctx=mx.current_context(), args=arg_array, aux_states=aux_array, grad_req='null')
    partitioned_exe = partitioned_sym.bind(ctx=mx.current_context(), args=arg_array,
                                           aux_states=aux_array, grad_req='null')
    exe.forward()
    partitioned_exe.forward()
    assert len(exe.outputs) == len(partitioned_exe.outputs)
    for i in range(len(exe.outputs)):
        assert_almost_equal((exe.outputs[i] - partitioned_exe.outputs[i]).abs().sum().asnumpy(),
                            np.zeros(shape=(1,)))

def check_subgraph_exe4(sym, subgraph_backend, op_names):
    """Use env var MXNET_SUBGRAPH_BACKEND=default to trigger graph partitioning in bind
    and compare results of the partitioned sym and the original sym."""
    def get_executor(sym, subgraph_backend=None, op_names=None, original_exec=None):
        if subgraph_backend is not None:
            os.environ['MXNET_SUBGRAPH_BACKEND'] = subgraph_backend
            check_call(_LIB.MXSetSubgraphPropertyOpNames(c_str(subgraph_backend), mx_uint(len(op_names)),
                                                         c_str_array(op_names)))
        arg_shapes, _, aux_shapes = sym.infer_shape()
        if subgraph_backend is None:
            arg_array = [mx.nd.random.uniform(shape=shape) for shape in arg_shapes]
            aux_array = [mx.nd.random.uniform(shape=shape) for shape in aux_shapes]
        else:
            arg_array = None
            aux_array = None
        exe = sym.bind(ctx=mx.current_context(),
                       args=arg_array if subgraph_backend is None else original_exec.arg_arrays,
                       aux_states=aux_array if subgraph_backend is None else original_exec.aux_arrays,
                       grad_req='null')
        exe.forward()
        if subgraph_backend is not None:
            check_call(_LIB.MXRemoveSubgraphPropertyOpNames(c_str(subgraph_backend)))
            del os.environ['MXNET_SUBGRAPH_BACKEND']
        return exe

    original_exec = get_executor(sym)
    partitioned_exec = get_executor(sym, subgraph_backend, op_names, original_exec)
    outputs1 = original_exec.outputs
    outputs2 = partitioned_exec.outputs
    assert len(outputs1) == len(outputs2)
    for i in range(len(outputs1)):
        assert_almost_equal((outputs1[i] - outputs2[i]).abs().sum().asnumpy(), np.zeros(shape=(1,)))

def set_random_inputs(exe1, input_names):
    """Sets random values to exe1's args and auxs"""
    for name in input_names:
        if name in exe1.arg_dict:
            exe1.arg_dict[name][:] = mx.nd.random.uniform(shape=exe1.arg_dict[name].shape)
        else:
            assert name in exe1.aux_dict
            exe1.aux_dict[name][:] = mx.nd.random.uniform(shape=exe1.aux_dict[name].shape)

def copy_inputs_between_executors(exe1, exe2, input_names):
    """Copies values of args and auxs from exe1 to exe2"""
    for name in input_names:
        if name in exe2.arg_dict:
            exe2.arg_dict[name][:] = exe1.arg_dict[name]
        else:
            assert name in exe2.aux_dict
            exe2.aux_dict[name][:] = exe1.aux_dict[name]

def check_subgraph_exe5(sym, subgraph_backend, op_names):
    """Call optimize_for to trigger graph partitioning without infer shapes/types before,
    then simple_bind and compare results of the partitioned sym and the original sym."""
    # simple_bind
    exe1 = sym.simple_bind(ctx=mx.current_context(), grad_req='null')
    input_names = sym.list_inputs()
    set_random_inputs(exe1, input_names)
    exe1.forward()

    # partition before simple_bind
    check_call(_LIB.MXSetSubgraphPropertyOpNamesV2(c_str(subgraph_backend), mx_uint(len(op_names)),
                                                 c_str_array(op_names)))
    part_sym = sym.optimize_for(subgraph_backend)
    check_call(_LIB.MXRemoveSubgraphPropertyOpNamesV2(c_str(subgraph_backend)))

    exe2 = part_sym.simple_bind(ctx=mx.current_context(), grad_req='null')
    copy_inputs_between_executors(exe1, exe2, input_names)
    exe2.forward()

    # compare outputs
    outputs1 = exe1.outputs
    outputs2 = exe2.outputs
    assert len(outputs1) == len(outputs2)
    for i in range(len(outputs1)):
        assert_almost_equal((outputs1[i] - outputs2[i]).abs().sum().asnumpy(), np.zeros(shape=(1,)))

def check_subgraph_exe6(sym, subgraph_backend, op_names):
    """Call optimize_for to trigger graph partitioning with shapes/types, then simple_bind 
    and compare results of the partitioned sym and the original sym."""
    # simple_bind
    exe1 = sym.simple_bind(ctx=mx.current_context(), grad_req='null')
    input_names = sym.list_inputs()
    set_random_inputs(exe1, input_names)
    exe1.forward()

    # infer shape/type before partition before simple_bind
    check_call(_LIB.MXSetSubgraphPropertyOpNamesV2(c_str(subgraph_backend), mx_uint(len(op_names)),
                                                 c_str_array(op_names)))
    part_sym = sym.optimize_for(subgraph_backend, exe1.arg_dict, exe1.aux_dict)
    check_call(_LIB.MXRemoveSubgraphPropertyOpNamesV2(c_str(subgraph_backend)))

    exe2 = part_sym.simple_bind(ctx=mx.current_context(), grad_req='null')
    copy_inputs_between_executors(exe1, exe2, input_names)
    exe2.forward()

    # compare outputs
    outputs1 = exe1.outputs
    outputs2 = exe2.outputs
    assert len(outputs1) == len(outputs2)
    for i in range(len(outputs1)):
        assert_almost_equal((outputs1[i] - outputs2[i]).abs().sum().asnumpy(), np.zeros(shape=(1,)))

def check_subgraph_exe7(sym, subgraph_backend, op_names):
    """Call optimize_for to trigger graph partitioning without infer shapes/types before,
    then bind and compare results of the partitioned sym and the original sym."""
    # bind
    arg_shapes, _, aux_shapes = sym.infer_shape()
    arg_array = [mx.nd.random.uniform(shape=shape) for shape in arg_shapes]
    aux_array = [mx.nd.random.uniform(shape=shape) for shape in aux_shapes]
    exe1 = sym.bind(ctx=mx.current_context(), args=arg_array, aux_states=aux_array, grad_req='null')
    exe1.forward()

    # partition before bind
    check_call(_LIB.MXSetSubgraphPropertyOpNamesV2(c_str(subgraph_backend), mx_uint(len(op_names)),
                                                 c_str_array(op_names)))
    part_sym = sym.optimize_for(subgraph_backend)
    check_call(_LIB.MXRemoveSubgraphPropertyOpNamesV2(c_str(subgraph_backend)))

    exe2 = part_sym.bind(ctx=mx.current_context(), args=arg_array, aux_states=aux_array, grad_req='null')
    exe2.forward()

    # compare outputs
    outputs1 = exe1.outputs
    outputs2 = exe2.outputs
    assert len(outputs1) == len(outputs2)
    for i in range(len(outputs1)):
        assert_almost_equal((outputs1[i] - outputs2[i]).abs().sum().asnumpy(), np.zeros(shape=(1,)))

def check_subgraph_exe8(sym, subgraph_backend, op_names):
    """Call optimize_for to infer shapes, types and dtypes followed by graph partitioning,
    then bind and compare results of the partitioned sym and the original sym."""
    # bind
    arg_shapes, _, aux_shapes = sym.infer_shape()
    arg_array = [mx.nd.random.uniform(shape=shape) for shape in arg_shapes]
    aux_array = [mx.nd.random.uniform(shape=shape) for shape in aux_shapes]
    exe1 = sym.bind(ctx=mx.current_context(), args=arg_array, aux_states=aux_array, grad_req='null')
    exe1.forward()

    # infer shape/type before partition before bind
    check_call(_LIB.MXSetSubgraphPropertyOpNamesV2(c_str(subgraph_backend), mx_uint(len(op_names)),
                                                 c_str_array(op_names)))
    part_sym = sym.optimize_for(subgraph_backend, arg_array, aux_array)
    check_call(_LIB.MXRemoveSubgraphPropertyOpNamesV2(c_str(subgraph_backend)))

    exe2 = part_sym.bind(ctx=mx.current_context(), args=arg_array, aux_states=aux_array, grad_req='null')
    exe2.forward()
    
    # compare outputs
    outputs1 = exe1.outputs
    outputs2 = exe2.outputs
    assert len(outputs1) == len(outputs2)
    for i in range(len(outputs1)):
        assert_almost_equal((outputs1[i] - outputs2[i]).abs().sum().asnumpy(), np.zeros(shape=(1,)))
    
def check_subgraph_exe9(sym, subgraph_backend, op_names):
    """Call hybridize() to partition the graph, and then compare results of the partitioned 
    sym and the original sym. Here do an inference before hybridizing with the subgraph_backend 
    which means we'll pass shapes/types"""
    # create Gluon block for given symbol
    inputs = [mx.sym.var(i, dtype=mx_real_t) for i in sym[1]]
    sym_block = nn.SymbolBlock(sym[0], inputs)
    sym_block.initialize(ctx=mx.current_context())
    x = [mx.nd.random.uniform(shape=s,ctx=mx.current_context()) for s in sym[2]]
    # hybridize and export to get baseline
    sym_block.hybridize()
    outputs1 = sym_block(*x)
    sym_block.export('check_subgraph_exe9')

    # load model and partition
    sym_block = nn.SymbolBlock.imports('check_subgraph_exe9-symbol.json',sym[1], 'check_subgraph_exe9-0000.params',
                                       ctx=mx.current_context())
    check_call(_LIB.MXSetSubgraphPropertyOpNamesV2(c_str(subgraph_backend), mx_uint(len(op_names)),
                                                c_str_array(op_names)))
    sym_block.hybridize(backend=subgraph_backend)
    outputs2 = sym_block(*x)
    check_call(_LIB.MXRemoveSubgraphPropertyOpNamesV2(c_str(subgraph_backend)))

    # compare outputs
    assert len(outputs1) == len(outputs2)
    for i in range(len(outputs1)):
        assert_almost_equal((outputs1[i] - outputs2[i]).abs().sum().asnumpy(), np.zeros(shape=(1,)))

def check_subgraph(subgraph_backend):
    for sym, op_names in get_graphs():
        check_subgraph_exe1(sym[0], subgraph_backend, op_names)
        check_subgraph_exe2(sym[0], subgraph_backend, op_names)
        check_subgraph_exe3(sym[0], subgraph_backend, op_names)
        check_subgraph_exe4(sym[0], subgraph_backend, op_names)

def check_subgraph_backend_sym(subgraph_backend):
    for sym, op_names in get_graphs():
        check_subgraph_exe5(sym[0], subgraph_backend, op_names)
        check_subgraph_exe6(sym[0], subgraph_backend, op_names)
        check_subgraph_exe7(sym[0], subgraph_backend, op_names)
        check_subgraph_exe8(sym[0], subgraph_backend, op_names)

def check_subgraph_backend_gluon(subgraph_backend):
    for sym, op_names in get_graphs():
        check_subgraph_exe9(sym, subgraph_backend, op_names)

# Test graph partition for 'default' backend.
def test_subgraph():
    check_subgraph('default')

# Test graph partition for 'default_v2' backend.
def test_subgraph_v2():
    check_subgraph('default_v2')

# Test enhanced Python and C APIs for graph partitioning given 'default' backend.
def test_subgraph_backend_sym():
    check_subgraph_backend_sym('default')

# Test enhanced Python and C APIs for graph partitioning given 'default_v2' backend.
def test_subgraph_backend_sym_v2():
    check_subgraph_backend_sym('default_v2')

# Test Gluon HybridBlocks for graph partitioning given 'default' backend.
def test_subgraph_backend_gluon():
    check_subgraph_backend_gluon('default')

# Test Gluon HybridBlocks for graph partitioning given 'default_v2' backend.
def test_subgraph_backend_gluon_v2():
    check_subgraph_backend_gluon('default_v2')

# Test Gluon HybridBlocks for graph partitioning a network created by HybridSequential.
def test_subgraph_backend_gluon_ext1():
    def get_net():
        net = nn.HybridSequential()  # Here we use the class HybridSequential.
        net.add(nn.Dense(256, activation='relu'),
                nn.Dense(128, activation='relu'),
                nn.Dense(2))
        return net

    # regular inference
    x = nd.random.normal(shape=(1, 512),ctx=mx.current_context())
    net = get_net()
    net.collect_params().initialize(ctx=mx.current_context())
    outputs1 = net(x)
    net.save_parameters('test_subgraph_backend_gluon_ext1.params')

    # after partitioning
    net = get_net()
    net.load_parameters('test_subgraph_backend_gluon_ext1.params',ctx=mx.current_context())
    subgraph_backend = 'default'
    op_names = ['FullyConnected']
    check_call(_LIB.MXSetSubgraphPropertyOpNamesV2(c_str(subgraph_backend), mx_uint(len(op_names)),
                                                c_str_array(op_names)))
    net.hybridize(backend = subgraph_backend)
    outputs2 = net(x)
    check_call(_LIB.MXRemoveSubgraphPropertyOpNamesV2(c_str(subgraph_backend)))

    # compare outputs
    assert len(outputs1) == len(outputs2)
    for i in range(len(outputs1)):
        assert_almost_equal((outputs1[i] - outputs2[i]).abs().sum().asnumpy(), np.zeros(shape=(1,)))

# Test Gluon HybridBlocks for graph partitioning a network created by HybridBlock.
def test_subgraph_backend_gluon_ext2():
    class Net(gluon.HybridBlock):
        def __init__(self, **kwargs):
            super(Net, self).__init__(**kwargs)
            with self.name_scope():
                self.fc1 = nn.Dense(256)
                self.fc2 = nn.Dense(128)
                self.fc3 = nn.Dense(2)

        def hybrid_forward(self, F, x):
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            return self.fc3(x)
    # regular inference
    x = nd.random.normal(shape=(1, 512),ctx=mx.current_context())
    net = Net()
    net.collect_params().initialize(ctx=mx.current_context())
    outputs1 = net(x)
    net.save_parameters('test_subgraph_backend_gluon_ext2.params')

    # after partitioning
    net = Net()
    net.load_parameters('test_subgraph_backend_gluon_ext2.params',ctx=mx.current_context())
    subgraph_backend = 'default'
    op_names = ['FullyConnected']
    check_call(_LIB.MXSetSubgraphPropertyOpNamesV2(c_str(subgraph_backend), mx_uint(len(op_names)),
                                                c_str_array(op_names)))
    net.hybridize(backend = subgraph_backend)
    outputs2 = net(x)
    check_call(_LIB.MXRemoveSubgraphPropertyOpNamesV2(c_str(subgraph_backend)))

    # compare outputs
    assert len(outputs1) == len(outputs2)
    for i in range(len(outputs1)):
        assert_almost_equal((outputs1[i] - outputs2[i]).abs().sum().asnumpy(), np.zeros(shape=(1,)))

if __name__ == '__main__':
    import nose
    nose.runmodule()
