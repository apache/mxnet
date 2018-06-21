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
import mxnet as mx
from mxnet.base import SymbolHandle, check_call, _LIB, mx_uint, c_str_array
from mxnet.symbol import Symbol
import numpy as np


def test_subgraph():
    def get_graph():
        data1 = mx.sym.Variable('data1', shape=(3, 3, 10, 10), dtype=np.float32)
        data2 = mx.sym.Variable('data2', shape=(1, 0, 2, 2))
        data3 = mx.sym.sin(data2)
        conv = mx.sym.Convolution(data=data1, weight=data3, kernel=(2, 2), num_filter=1)
        rets = []
        rets.append((conv, []))
        rets.append((conv, [mx.sym.sin.__name__]))
        rets.append((conv, [mx.sym.Convolution.__name__]))
        rets.append((conv, [mx.sym.sin.__name__, mx.sym.Convolution.__name__]))
        return rets

    for regular_sym, op_names in get_graph():
        input_names = regular_sym.list_inputs()
        shapes = regular_sym.infer_shape()
        types = regular_sym.infer_type()
        out = SymbolHandle()

        check_call(_LIB.MXPartitionGraph(regular_sym.handle, mx_uint(len(op_names)),
            c_str_array(op_names), ctypes.byref(out)))
        subgraph_sym = Symbol(out)
        assert input_names == subgraph_sym.list_inputs()

        print(subgraph_sym.list_outputs())
        assert shapes == subgraph_sym.infer_shape()
        assert types == subgraph_sym.infer_type()

        regular_exec = regular_sym.simple_bind(ctx=mx.cpu(), grad_req='null')
        subgraph_exec = subgraph_sym.simple_bind(ctx=mx.cpu(), grad_req='null')

        for name in input_names:
            regular_exec.arg_dict[name][:] = mx.nd.random.normal(
                    shape=regular_exec.arg_dict[name].shape)
            subgraph_exec.arg_dict[name][:] = regular_exec.arg_dict[name]

        subgraph_exec.forward()
        regular_exec.forward()
        mx.nd.waitall()
        assert (subgraph_exec.outputs[0] - regular_exec.outputs[0]).abs().sum().asscalar() == 0.0


def test_input_name_order():
    def check_input_order(sym, op_names):
        out = SymbolHandle()
        check_call(_LIB.MXPartitionGraph(sym.handle, mx_uint(len(op_names)),
                                         c_str_array(op_names), ctypes.byref(out)))

        new_sym = Symbol(out)
        #print(sym.list_inputs())
        #print(new_sym.list_inputs())
        assert new_sym.list_inputs() == sym.list_inputs()
        assert new_sym.list_arguments() == sym.list_arguments()
        assert new_sym.list_auxiliary_states() == sym.list_auxiliary_states()
        #print(new_sym.list_arguments())
        #print(new_sym.list_auxiliary_states())
        #print('original outputs: %s' % sym.list_outputs())
        #print('new sym outputs: %s' % new_sym.list_outputs())

    def test_network_structure_1():
        data1 = mx.sym.var('data1')
        data2 = mx.sym.var('data2')
        conv1 = mx.sym.Convolution(data=data1, weight=data2, no_bias=True, kernel=(2, 2), num_filter=1)
        conv2 = mx.sym.Convolution(data=data2, weight=data1, no_bias=True, kernel=(2, 2), num_filter=1)
        out = mx.sym.Group([conv1, conv2])
        check_input_order(out, ['Convolution'])

    def test_network_structure_2():
        data1 = mx.sym.var('data1')
        data2 = mx.sym.var('data2')
        conv1 = mx.sym.Convolution(data=data1, weight=data2, no_bias=True, kernel=(2, 2), num_filter=1)
        conv2 = mx.sym.Convolution(data=data2, weight=data1, no_bias=True, kernel=(2, 2), num_filter=1)
        out = conv1 + conv2
        check_input_order(out, ['Convolution'])
        check_input_order(out, ['Convolution', '_Plus', 'elemwise_add', '_plus'])

    def test_network_structure_3():
        # this tests whether the partitioning algorithm can deal with cycles
        data = mx.sym.var('data')
        ret = mx.sym.exp(data)
        ret1 = mx.sym.cos(ret)
        ret2 = mx.sym.sin(ret)
        ret = ret1 + ret2
        check_input_order(ret, ['exp', 'sin', '_Plus', 'elemwise_add', '_plus'])
        check_input_order(ret, ['exp', 'cos', '_Plus', 'elemwise_add', '_plus'])

    def test_network_structure_4():
        # this tests whether the partitioned sym can distinguish in_args and aux_states
        data = mx.sym.var('data')
        ret = mx.sym.exp(data)
        ret1 = mx.sym.cos(ret)
        ret2 = mx.sym.sin(ret)
        ret = ret1 + ret2
        ret = mx.sym.BatchNorm(ret)
        ret = mx.sym.BatchNorm(ret)
        check_input_order(ret, ['exp', 'sin', '_Plus', 'elemwise_add', '_plus'])
        check_input_order(ret, ['exp', 'cos', '_Plus', 'elemwise_add', '_plus'])
        check_input_order(ret, ['exp', 'sin', '_Plus', 'elemwise_add', '_plus', 'BatchNorm'])
        check_input_order(ret, ['exp', 'cos', '_Plus', 'elemwise_add', '_plus', 'BatchNorm'])
        check_input_order(ret, ['exp', 'BatchNorm'])
        check_input_order(ret, ['BatchNorm'])

    test_network_structure_1()
    test_network_structure_2()
    test_network_structure_3()
    test_network_structure_4()


if __name__ == '__main__':
    import nose
    nose.runmodule()
