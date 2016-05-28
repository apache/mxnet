import sys
import os
curr_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
sys.path.insert(0, os.path.join(curr_path, '../unittest'))
from test_operator import *
import mxnet as mx
import numpy as np
from numpy.testing import assert_allclose
import time

def check_consistency(sym, ctx_list, scale=1.0):
    tol = {np.dtype(np.float16): 1e-1,
           np.dtype(np.float32): 1e-3,
           np.dtype(np.float64): 1e-5,
           np.dtype(np.uint8): 0,
           np.dtype(np.int32): 0}
    assert(len(ctx_list) > 1)
    exe_list = [sym.simple_bind(grad_req='write', **ctx) for ctx in ctx_list]
    for exe in exe_list:
        assert(len(exe.outputs) == 1)
        assert(len(exe.arg_arrays) == len(exe_list[0].arg_arrays))
        assert(len(exe.grad_arrays) == len(exe_list[0].grad_arrays))

    init = [np.random.normal(size=arr.shape, scale=scale) for arr in exe_list[0].arg_arrays]
    for exe in exe_list:
        for arr, iarr in zip(exe.arg_arrays, init):
            arr[:] = iarr.astype(arr.dtype)

    # forward
    for exe in exe_list:
        exe.forward(is_train=True)
        exe.backward(exe.outputs[0])

    outputs = [exe.outputs[0].asnumpy() for exe in exe_list]
    grads = [[grad.asnumpy() for grad in exe.grad_arrays] for exe in exe_list]
    dtypes = [arr.dtype for arr in outputs]
    max_idx = np.argmax(dtypes)

    for i, exe in enumerate(exe_list):
        if i == max_idx:
            continue
        for arr1, arr2 in zip([outputs[i]]+grads[i], [outputs[max_idx]]+grads[max_idx]):
            arr2 = arr2.astype(dtypes[i])
            try:
                assert_allclose(arr1, arr2, rtol=tol[dtypes[i]], atol=tol[dtypes[i]])
            except Exception, e:
                print e

def check_speed(sym, ctx, scale=1.0, N=100):
    exe = sym.simple_bind(grad_req='write', **ctx)
    init = [np.random.normal(size=arr.shape, scale=scale) for arr in exe.arg_arrays]
    for arr, iarr in zip(exe.arg_arrays, init):
        arr[:] = iarr.astype(arr.dtype)

    # warm up
    exe.forward(is_train=True)
    exe.backward(exe.outputs[0])
    exe.outputs[0].wait_to_read()

    tic = time.time()
    for i in range(N):
        exe.forward(is_train=True)
        exe.backward(exe.outputs[0])
        exe.outputs[0].wait_to_read()
    return (time.time() - tic)*1.0/N




def test_convolution_with_type():
    sym = mx.sym.Convolution(num_filter=3, kernel=(3,3), name='conv')
    ctx_list = [{'ctx': mx.gpu(0), 'conv_data': (2, 2, 10, 10), 'type_dict': {'conv_data': np.float64}},
                {'ctx': mx.gpu(0), 'conv_data': (2, 2, 10, 10), 'type_dict': {'conv_data': np.float32}},
                {'ctx': mx.gpu(0), 'conv_data': (2, 2, 10, 10), 'type_dict': {'conv_data': np.float16}},
                {'ctx': mx.cpu(0), 'conv_data': (2, 2, 10, 10), 'type_dict': {'conv_data': np.float64}},
                {'ctx': mx.cpu(0), 'conv_data': (2, 2, 10, 10), 'type_dict': {'conv_data': np.float32}}]
    check_consistency(sym, ctx_list)

def test_fullyconnected_with_type():
    sym = mx.sym.FullyConnected(num_hidden=3, name='inner')
    ctx_list = [{'ctx': mx.gpu(0), 'inner_data': (2, 10), 'type_dict': {'inner_data': np.float64}},
                {'ctx': mx.gpu(0), 'inner_data': (2, 10), 'type_dict': {'inner_data': np.float32}},
                {'ctx': mx.gpu(0), 'inner_data': (2, 10), 'type_dict': {'inner_data': np.float16}},
                {'ctx': mx.cpu(0), 'inner_data': (2, 10), 'type_dict': {'inner_data': np.float64}},
                {'ctx': mx.cpu(0), 'inner_data': (2, 10), 'type_dict': {'inner_data': np.float32}}]
    check_consistency(sym, ctx_list)

def test_activation_with_type():
    sym = mx.sym.Activation(name='act', act_type='sigmoid')
    ctx_list = [{'ctx': mx.gpu(0), 'act_data': (2, 2, 10, 10), 'type_dict': {'act_data': np.float64}},
                {'ctx': mx.gpu(0), 'act_data': (2, 2, 10, 10), 'type_dict': {'act_data': np.float32}},
                {'ctx': mx.gpu(0), 'act_data': (2, 2, 10, 10), 'type_dict': {'act_data': np.float16}},
                {'ctx': mx.cpu(0), 'act_data': (2, 2, 10, 10), 'type_dict': {'act_data': np.float64}},
                {'ctx': mx.cpu(0), 'act_data': (2, 2, 10, 10), 'type_dict': {'act_data': np.float32}},
                {'ctx': mx.cpu(0), 'act_data': (2, 2, 10, 10), 'type_dict': {'act_data': np.float16}}]
    check_consistency(sym, ctx_list)

if __name__ == '__main__':
    test_convolution_with_type()
    test_fullyconnected_with_type()
    test_activation_with_type()
	#test_softmax_with_shape((3,4), mx.gpu())
    #test_multi_softmax_with_shape((3,4,5), mx.gpu())