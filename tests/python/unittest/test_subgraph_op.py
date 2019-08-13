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
from mxnet.base import SymbolHandle, check_call, _LIB, mx_uint, c_str_array, c_str
from mxnet.symbol import Symbol
import numpy as np
from mxnet.test_utils import assert_almost_equal


def _test_subgraph_exe(subgraph_backend):
    def _check_subgraph_exe1(sym, subgraph_backend, op_names):
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

    def _check_subgraph_exe2(sym, subgraph_backend, op_names):
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

    def _check_subgraph_exe3(sym, subgraph_backend, op_names):
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

    def _check_subgraph_exe4(sym, subgraph_backend, op_names):
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

    def check_subgraph_exe(sym, subgraph_backend, op_names):
        _check_subgraph_exe1(sym, subgraph_backend, op_names)
        _check_subgraph_exe2(sym, subgraph_backend, op_names)
        _check_subgraph_exe3(sym, subgraph_backend, op_names)
        _check_subgraph_exe4(sym, subgraph_backend, op_names)

    def test_network_structure_1(subgraph_backend):
        data1 = mx.sym.var('data1', shape=(2, 3, 10, 10))
        data2 = mx.sym.var('data2')
        conv1 = mx.sym.Convolution(data=data1, weight=data2, no_bias=True, kernel=(2, 2), num_filter=1)
        conv2 = mx.sym.Convolution(data=data2, no_bias=True, kernel=(1, 1), num_filter=1)
        out = mx.sym.Group([conv1, conv2])
        check_subgraph_exe(out, subgraph_backend, ['Convolution'])

    def test_network_structure_2(subgraph_backend):
        # this tests whether the partitioning algorithm can deal with cycles
        data = mx.sym.var('data', shape=(2, 3, 10, 10))
        ret = mx.sym.exp(data)
        ret1 = mx.sym.cos(ret)
        ret2 = mx.sym.sin(ret)
        ret = ret1 + ret2
        check_subgraph_exe(ret, subgraph_backend, ['exp', 'sin', '_Plus', 'elemwise_add', '_plus'])
        check_subgraph_exe(ret, subgraph_backend, ['exp', 'cos', '_Plus', 'elemwise_add', '_plus'])

    def test_network_structure_3(subgraph_backend):
        # this tests whether the partitioned sym can distinguish in_args and aux_states
        data = mx.sym.var('data', shape=(2, 3, 10, 10))
        ret = mx.sym.exp(data)
        ret1 = mx.sym.cos(ret)
        ret2 = mx.sym.sin(ret)
        ret = ret1 + ret2
        ret = mx.sym.BatchNorm(ret)
        ret = mx.sym.BatchNorm(ret)
        check_subgraph_exe(ret, subgraph_backend, ['exp', 'sin', '_Plus', 'elemwise_add', '_plus'])
        check_subgraph_exe(ret, subgraph_backend, ['exp', 'cos', '_Plus', 'elemwise_add', '_plus'])
        check_subgraph_exe(ret, subgraph_backend, ['exp', 'sin', '_Plus', 'elemwise_add', '_plus', 'BatchNorm'])
        check_subgraph_exe(ret, subgraph_backend, ['exp', 'cos', '_Plus', 'elemwise_add', '_plus', 'BatchNorm'])
        check_subgraph_exe(ret, subgraph_backend, ['exp', 'BatchNorm'])
        check_subgraph_exe(ret, subgraph_backend, ['BatchNorm'])

    def test_network_structure_4(subgraph_backend):
        # the last op has multiple duplicate outputs
        data = mx.sym.var('data', shape=(2, 3, 10, 10))
        ret = mx.sym.exp(data)
        ret = mx.sym.Group([ret, ret, ret])
        check_subgraph_exe(ret, subgraph_backend, ['exp'])

    def test_network_structure_5(subgraph_backend):
        # the subgraph has two duplicate input entries
        data = mx.sym.var('data', shape=(2, 3, 10, 10))
        ret = data + data
        check_subgraph_exe(ret, subgraph_backend, ['_plus', '_Plus', 'elemwise_add'])

    def test_network_structure_6(subgraph_backend):
        def get_graph():
            data1 = mx.sym.Variable('data1', shape=(3, 3, 10, 10), dtype=np.float32)
            data2 = mx.sym.Variable('data2', shape=(1, 0, 2, 2))
            data3 = mx.sym.sin(data2)
            conv = mx.sym.Convolution(data=data1, weight=data3, kernel=(2, 2), num_filter=1)
            rets = [(conv, []),
                    (conv, [mx.sym.sin.__name__]),
                    (conv, [mx.sym.Convolution.__name__]),
                    (conv, [mx.sym.sin.__name__, mx.sym.Convolution.__name__])]
            return rets

        for sym, op_names in get_graph():
            check_subgraph_exe(sym, subgraph_backend, op_names)

    def test_network_structure_7(subgraph_backend):
        # in this graph, the subgraph node and the other two external nodes form a cycle
        data = mx.sym.Variable('data', shape=(1,))
        ret1 = mx.sym.sin(data)
        ret2 = mx.sym.cos(ret1)
        for _ in range(5):
            ret2 = mx.sym.cos(ret2)
        ret = ret1 + ret2
        check_subgraph_exe(ret, subgraph_backend, ['sin', 'elemwise_add', '_plus', '_Plus'])

    test_network_structure_1(subgraph_backend)
    test_network_structure_2(subgraph_backend)
    test_network_structure_3(subgraph_backend)
    test_network_structure_4(subgraph_backend)
    test_network_structure_5(subgraph_backend)
    test_network_structure_6(subgraph_backend)
    test_network_structure_7(subgraph_backend)

def test_subgraph_exe():
    _test_subgraph_exe('default')

def test_subgraph_v2_exe():
    _test_subgraph_exe('default_v2')

if __name__ == '__main__':
    import nose
    nose.runmodule()
