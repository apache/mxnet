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
from mxnet.test_utils import assert_almost_equal


def test_subgraph_exe():
    def check_subgraph_exe(sym, op_names):
        out = SymbolHandle()
        check_call(_LIB.MXPartitionGraph(sym.handle, mx_uint(len(op_names)),
                                         c_str_array(op_names), ctypes.byref(out)))

        partitioned_sym = Symbol(out)
        assert partitioned_sym.list_inputs() == sym.list_inputs()
        assert partitioned_sym.list_arguments() == sym.list_arguments()
        assert partitioned_sym.list_auxiliary_states() == sym.list_auxiliary_states()
        exe = sym.simple_bind(ctx=mx.cpu(), grad_req='null')
        partitioned_exe = partitioned_sym.simple_bind(ctx=mx.cpu(), grad_req='null')
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
        mx.nd.waitall()
        assert len(exe.outputs) == len(partitioned_exe.outputs)
        for i in range(len(exe.outputs)):
            assert_almost_equal((exe.outputs[i] - partitioned_exe.outputs[i]).abs().sum().asnumpy(),
                                np.zeros(shape=(1,)))

    def test_network_structure_1():
        data1 = mx.sym.var('data1', shape=(2, 3, 10, 10))
        data2 = mx.sym.var('data2')
        conv1 = mx.sym.Convolution(data=data1, weight=data2, no_bias=True, kernel=(2, 2), num_filter=1)
        conv2 = mx.sym.Convolution(data=data2, no_bias=True, kernel=(1, 1), num_filter=1)
        out = mx.sym.Group([conv1, conv2])
        check_subgraph_exe(out, ['Convolution'])

    def test_network_structure_2():
        # this tests whether the partitioning algorithm can deal with cycles
        data = mx.sym.var('data', shape=(2, 3, 10, 10))
        ret = mx.sym.exp(data)
        ret1 = mx.sym.cos(ret)
        ret2 = mx.sym.sin(ret)
        ret = ret1 + ret2
        check_subgraph_exe(ret, ['exp', 'sin', '_Plus', 'elemwise_add', '_plus'])
        check_subgraph_exe(ret, ['exp', 'cos', '_Plus', 'elemwise_add', '_plus'])

    def test_network_structure_3():
        # this tests whether the partitioned sym can distinguish in_args and aux_states
        data = mx.sym.var('data', shape=(2, 3, 10, 10))
        ret = mx.sym.exp(data)
        ret1 = mx.sym.cos(ret)
        ret2 = mx.sym.sin(ret)
        ret = ret1 + ret2
        ret = mx.sym.BatchNorm(ret)
        ret = mx.sym.BatchNorm(ret)
        check_subgraph_exe(ret, ['exp', 'sin', '_Plus', 'elemwise_add', '_plus'])
        check_subgraph_exe(ret, ['exp', 'cos', '_Plus', 'elemwise_add', '_plus'])
        check_subgraph_exe(ret, ['exp', 'sin', '_Plus', 'elemwise_add', '_plus', 'BatchNorm'])
        check_subgraph_exe(ret, ['exp', 'cos', '_Plus', 'elemwise_add', '_plus', 'BatchNorm'])
        check_subgraph_exe(ret, ['exp', 'BatchNorm'])
        check_subgraph_exe(ret, ['BatchNorm'])

    def test_network_structure_4():
        # the last op has multiple duplicate outputs
        data = mx.sym.var('data', shape=(2, 3, 10, 10))
        ret = mx.sym.exp(data)
        ret = mx.sym.Group([ret, ret, ret])
        check_subgraph_exe(ret, ['exp'])

    def test_network_structure_5():
        # the subgraph has two duplicate input entries
        data = mx.sym.var('data', shape=(2, 3, 10, 10))
        ret = data + data
        check_subgraph_exe(ret, ['_plus', '_Plus', 'elemwise_add'])

    def test_network_structure_6():
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
            check_subgraph_exe(sym, op_names)

    def test_network_structure_7():
        # in this graph, the subgraph node and the other two external nodes form a cycle
        data = mx.sym.Variable('data', shape=(1,))
        ret1 = mx.sym.sin(data)
        ret2 = mx.sym.cos(ret1)
        for _ in range(5):
            ret2 = mx.sym.cos(ret2)
        ret = ret1 + ret2
        check_subgraph_exe(ret, ['sin', 'elemwise_add', '_plus', '_Plus'])

    test_network_structure_1()
    test_network_structure_2()
    test_network_structure_3()
    test_network_structure_4()
    test_network_structure_5()
    test_network_structure_6()
    test_network_structure_7()


if __name__ == '__main__':
    import nose
    nose.runmodule()
