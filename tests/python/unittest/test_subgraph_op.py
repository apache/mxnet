import ctypes
import mxnet as mx
from mxnet.base import SymbolHandle, check_call, _LIB, mx_uint, c_str_array
from mxnet.symbol import Symbol
import numpy as np


def test_subgraph_op_whole_graph():
    data1 = mx.sym.Variable('data1', shape=(3, 3, 10, 10), dtype=np.float32)
    data2 = mx.sym.Variable('data2', shape=(1, 0, 2, 2))
    data3 = mx.sym.sin(data2)

    regular_sym = mx.sym.Convolution(data=data1, weight=data3, kernel=(2, 2), num_filter=1)

    out = SymbolHandle()

    op_names = []
    op_names = [mx.sym.sin.__name__, mx.sym.Convolution.__name__]

    check_call(_LIB.MXPartitionGraph(regular_sym.handle, mx_uint(len(op_names)),
                                     c_str_array(op_names), ctypes.byref(out)))

    subgraph_sym = Symbol(out)
    assert regular_sym.list_inputs() == subgraph_sym.list_inputs()
    input_names = subgraph_sym.list_inputs()

    #assert regular_sym.list_outputs() == subgraph_sym.list_outputs()
    print(subgraph_sym.list_outputs())
    assert regular_sym.infer_shape() == subgraph_sym.infer_shape()
    assert regular_sym.infer_type() == subgraph_sym.infer_type()

    regular_exec = regular_sym.simple_bind(ctx=mx.cpu(), grad_req='null')
    subgraph_exec = subgraph_sym.simple_bind(ctx=mx.cpu(), grad_req='null')

    regular_exec.arg_dict[data1.name][:] = mx.nd.random.normal(shape=regular_exec.arg_dict[data1.name].shape)
    regular_exec.arg_dict[data2.name][:] = mx.nd.random.normal(shape=regular_exec.arg_dict[data2.name].shape)
    regular_exec.arg_dict[input_names[-1]][:] = mx.nd.random.normal(shape=regular_exec.arg_dict[input_names[-1]].shape)

    subgraph_exec.arg_dict[data1.name][:] = regular_exec.arg_dict[data1.name]
    subgraph_exec.arg_dict[data2.name][:] = regular_exec.arg_dict[data2.name]
    subgraph_exec.arg_dict[input_names[-1]][:] = regular_exec.arg_dict[input_names[-1]]

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
        print('original outputs: %s' % sym.list_outputs())
        print('new sym outputs: %s' % new_sym.list_outputs())

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

    test_network_structure_1()
    test_network_structure_2()
    test_network_structure_3()


if __name__ == '__main__':
    import nose
    nose.runmodule()
