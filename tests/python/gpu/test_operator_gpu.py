import sys
import os
curr_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
sys.path.insert(0, os.path.join(curr_path, '../unittest'))
from test_operator import *
import mxnet as mx
import numpy as np
from numpy.testing import assert_allclose
import time

def check_consistency(sym, ctx_list, scale=1.0, grad_req='write'):
    tol = {np.dtype(np.float16): 1e-1,
           np.dtype(np.float32): 1e-3,
           np.dtype(np.float64): 1e-5,
           np.dtype(np.uint8): 0,
           np.dtype(np.int32): 0}
    assert(len(ctx_list) > 1)
    exe_list = [sym.simple_bind(grad_req=grad_req, **ctx) for ctx in ctx_list]
    for exe in exe_list:
        assert(len(exe.outputs) == 1)
        assert(len(exe.arg_arrays) == len(exe_list[0].arg_arrays))
        assert(len(exe.grad_arrays) == len(exe_list[0].grad_arrays))

    init = [np.random.normal(size=arr.shape, scale=scale) for arr in exe_list[0].arg_arrays]
    if sym.name == 'embedding':
        init[0] = np.random.randint(low=0, high=10, size=exe_list[0].arg_arrays[0].shape)

    for exe in exe_list:
        for arr, iarr in zip(exe.arg_arrays, init):
            arr[:] = iarr.astype(arr.dtype)

    # forward
    for exe in exe_list:
        exe.forward(is_train=True)
        exe.backward(exe.outputs[0])

    outputs = [exe.outputs[0].asnumpy() for exe in exe_list]
    # lazy solution handling None grad
    grads = [[grad.asnumpy() if grad is not None else np.zeros(1) for grad in exe.grad_arrays] for exe in exe_list]
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

    #forward predict
    for exe in exe_list:
        exe.forward(is_train=False)

    outputs = [exe.outputs[0].asnumpy() for exe in exe_list]
    dtypes = [arr.dtype for arr in outputs]
    max_idx = np.argmax(dtypes)

    for i, exe in enumerate(exe_list):
        if i == max_idx:
            continue
        for arr1, arr2 in zip([outputs[i]], [outputs[max_idx]]):
            arr2 = arr2.astype(dtypes[i])
            try:
                assert_allclose(arr1, arr2, rtol=tol[dtypes[i]], atol=tol[dtypes[i]])
            except Exception, e:
                print e

def check_speed(sym, ctx, scale=1.0, N=100, grad_req='write'):
    exe = sym.simple_bind(grad_req=grad_req, **ctx)
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

def test_batchnorm_with_type():
    sym = mx.sym.BatchNorm(name='norm', fix_gamma=False)
    ctx_list = [{'ctx': mx.gpu(0), 'norm_data': (10, 2, 10, 10), 'type_dict': {'norm_data': np.float32}},
                {'ctx': mx.cpu(0), 'norm_data': (10, 2, 10, 10), 'type_dict': {'norm_data': np.float32}}]
    check_consistency(sym, ctx_list)

    sym = mx.sym.BatchNorm(name='norm', fix_gamma=True)
    check_consistency(sym, ctx_list)

def test_convolution_with_type():
    sym = mx.sym.Convolution(num_filter=3, kernel=(3,3), name='conv')
    ctx_list = [{'ctx': mx.gpu(0), 'conv_data': (2, 2, 10, 10), 'type_dict': {'conv_data': np.float64}},
                {'ctx': mx.gpu(0), 'conv_data': (2, 2, 10, 10), 'type_dict': {'conv_data': np.float32}},
                {'ctx': mx.gpu(0), 'conv_data': (2, 2, 10, 10), 'type_dict': {'conv_data': np.float16}},
                {'ctx': mx.cpu(0), 'conv_data': (2, 2, 10, 10), 'type_dict': {'conv_data': np.float64}},
                {'ctx': mx.cpu(0), 'conv_data': (2, 2, 10, 10), 'type_dict': {'conv_data': np.float32}}]
    check_consistency(sym, ctx_list)

def test_deconvolution_with_type():
    sym = mx.sym.Deconvolution(num_filter=2, kernel=(3,3), name='deconv')
    ctx_list = [{'ctx': mx.gpu(0), 'deconv_data': (2, 2, 10, 10), 'type_dict': {'deconv_data': np.float64}},
                {'ctx': mx.gpu(0), 'deconv_data': (2, 2, 10, 10), 'type_dict': {'deconv_data': np.float32}},
                {'ctx': mx.gpu(0), 'deconv_data': (2, 2, 10, 10), 'type_dict': {'deconv_data': np.float16}},
                {'ctx': mx.cpu(0), 'deconv_data': (2, 2, 10, 10), 'type_dict': {'deconv_data': np.float64}},
                {'ctx': mx.cpu(0), 'deconv_data': (2, 2, 10, 10), 'type_dict': {'deconv_data': np.float32}}]
    check_consistency(sym, ctx_list)

def test_upsampling_with_type():
    sym = mx.sym.UpSampling(scale=2, num_filter=2, name='up', sample_type = 'nearest', num_args=1)
    ctx_list = [{'ctx': mx.gpu(0), 'up_arg0': (2, 2, 2, 10), 'type_dict': {'up_arg0': np.float64}},
                {'ctx': mx.gpu(0), 'up_arg0': (2, 2, 2, 10), 'type_dict': {'up_arg0': np.float32}},
                {'ctx': mx.gpu(0), 'up_arg0': (2, 2, 2, 10), 'type_dict': {'up_arg0': np.float16}},
                {'ctx': mx.cpu(0), 'up_arg0': (2, 2, 2, 10), 'type_dict': {'up_arg0': np.float64}},
                {'ctx': mx.cpu(0), 'up_arg0': (2, 2, 2, 10), 'type_dict': {'up_arg0': np.float32}}]
    check_consistency(sym, ctx_list)

def test_concat_with_type():
    sym = mx.sym.Concat(name='concat', num_args=2)
    ctx_list = [{'ctx': mx.gpu(0), 'concat_arg1': (2, 10), 'concat_arg0': (2, 10),
                 'type_dict': {'concat_arg0': np.float64, 'concat_arg1': np.float64}},
                {'ctx': mx.gpu(0), 'concat_arg1': (2, 10), 'concat_arg0': (2, 10),
                 'type_dict': {'concat_arg0': np.float32, 'concat_arg1': np.float32}},
                {'ctx': mx.gpu(0), 'concat_arg1': (2, 10), 'concat_arg0': (2, 10),
                 'type_dict': {'concat_arg0': np.float16, 'concat_arg1': np.float16}},
                {'ctx': mx.cpu(0), 'concat_arg1': (2, 10), 'concat_arg0': (2, 10),
                 'type_dict': {'concat_arg0': np.float64, 'concat_arg1': np.float64}},
                {'ctx': mx.cpu(0), 'concat_arg1': (2, 10), 'concat_arg0': (2, 10),
                 'type_dict': {'concat_arg0': np.float32, 'concat_arg1': np.float32}}]
    check_consistency(sym, ctx_list)

def test_elementwisesum_with_type():
    sym = mx.sym.ElementWiseSum(name='ews', num_args=2)
    ctx_list = [{'ctx': mx.gpu(0), 'ews_arg1': (2, 10), 'ews_arg0': (2, 10),
                 'type_dict': {'ews_arg0': np.float64, 'ews_arg1': np.float64}},
                {'ctx': mx.gpu(0), 'ews_arg1': (2, 10), 'ews_arg0': (2, 10),
                 'type_dict': {'ews_arg0': np.float32, 'ews_arg1': np.float32}},
                {'ctx': mx.gpu(0), 'ews_arg1': (2, 10), 'ews_arg0': (2, 10),
                 'type_dict': {'ews_arg0': np.float16, 'ews_arg1': np.float16}},
                {'ctx': mx.cpu(0), 'ews_arg1': (2, 10), 'ews_arg0': (2, 10),
                 'type_dict': {'ews_arg0': np.float64, 'ews_arg1': np.float64}},
                {'ctx': mx.cpu(0), 'ews_arg1': (2, 10), 'ews_arg0': (2, 10),
                 'type_dict': {'ews_arg0': np.float32, 'ews_arg1': np.float32}}]
    check_consistency(sym, ctx_list)


def test_reshape_with_type():
    sym = mx.sym.Reshape(name='reshape', shape=(-1,1,1,0))
    ctx_list = [{'ctx': mx.gpu(0), 'reshape_data': (2, 2, 2, 10), 'type_dict': {'reshape_data': np.float64}},
                {'ctx': mx.gpu(0), 'reshape_data': (2, 2, 2, 10), 'type_dict': {'reshape_data': np.float32}},
                {'ctx': mx.gpu(0), 'reshape_data': (2, 2, 2, 10), 'type_dict': {'reshape_data': np.float16}},
                {'ctx': mx.cpu(0), 'reshape_data': (2, 2, 2, 10), 'type_dict': {'reshape_data': np.float64}},
                {'ctx': mx.cpu(0), 'reshape_data': (2, 2, 2, 10), 'type_dict': {'reshape_data': np.float32}}]
    check_consistency(sym, ctx_list)

def test_blockgrad_with_type():
    sym = mx.sym.BlockGrad(name='bg')
    ctx_list = [{'ctx': mx.gpu(0), 'bg_data': (2, 2, 2, 10), 'type_dict': {'bg_data': np.float64}},
                {'ctx': mx.gpu(0), 'bg_data': (2, 2, 2, 10), 'type_dict': {'bg_data': np.float32}},
                {'ctx': mx.gpu(0), 'bg_data': (2, 2, 2, 10), 'type_dict': {'bg_data': np.float16}},
                {'ctx': mx.cpu(0), 'bg_data': (2, 2, 2, 10), 'type_dict': {'bg_data': np.float64}},
                {'ctx': mx.cpu(0), 'bg_data': (2, 2, 2, 10), 'type_dict': {'bg_data': np.float32}}]
    check_consistency(sym, ctx_list)

def test_swapaxis_with_type():
    sym = mx.sym.SwapAxis(name='swap', dim1=1)
    ctx_list = [{'ctx': mx.gpu(0), 'swap_data': (2, 2, 2, 10), 'type_dict': {'swap_data': np.float64}},
                {'ctx': mx.gpu(0), 'swap_data': (2, 2, 2, 10), 'type_dict': {'swap_data': np.float32}},
                {'ctx': mx.gpu(0), 'swap_data': (2, 2, 2, 10), 'type_dict': {'swap_data': np.float16}},
                {'ctx': mx.cpu(0), 'swap_data': (2, 2, 2, 10), 'type_dict': {'swap_data': np.float64}},
                {'ctx': mx.cpu(0), 'swap_data': (2, 2, 2, 10), 'type_dict': {'swap_data': np.float32}}]
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

def test_embedding_with_type():
    sym = mx.sym.Embedding(name='embedding', input_dim=10, output_dim=20)
    ctx_list = [{'ctx': mx.gpu(0), 'embedding_data': (2, 10), 'type_dict': {'embedding_data': np.float64}},
                {'ctx': mx.gpu(0), 'embedding_data': (2, 10), 'type_dict': {'embedding_data': np.float32}},
                {'ctx': mx.gpu(0), 'embedding_data': (2, 10), 'type_dict': {'embedding_data': np.float16}},
                {'ctx': mx.cpu(0), 'embedding_data': (2, 10), 'type_dict': {'embedding_data': np.float64}},
                {'ctx': mx.cpu(0), 'embedding_data': (2, 10), 'type_dict': {'embedding_data': np.float32}},
                {'ctx': mx.cpu(0), 'embedding_data': (2, 10), 'type_dict': {'embedding_data': np.float16}}]
    check_consistency(sym, ctx_list, grad_req={'embedding_data': 'null','embedding_weight': 'write'})

def test_pooling_with_type():
    sym= mx.sym.Pooling(name='pooling', kernel=(3, 3), pool_type='avg')
    ctx_list = [{'ctx': mx.gpu(0), 'pooling_data': (2, 2, 10, 10), 'type_dict': {'pooling_data': np.float64}},
                {'ctx': mx.gpu(0), 'pooling_data': (2, 2, 10, 10), 'type_dict': {'pooling_data': np.float32}},
                {'ctx': mx.gpu(0), 'pooling_data': (2, 2, 10, 10), 'type_dict': {'pooling_data': np.float16}},
                {'ctx': mx.cpu(0), 'pooling_data': (2, 2, 10, 10), 'type_dict': {'pooling_data': np.float64}},
                {'ctx': mx.cpu(0), 'pooling_data': (2, 2, 10, 10), 'type_dict': {'pooling_data': np.float32}}]
    check_consistency(sym, ctx_list)

    sym_3d= mx.sym.Pooling(name='pooling', kernel=(3, 3, 3), pool_type='avg')
    ctx_list_3d = [{'ctx': mx.gpu(0), 'pooling_data': (2, 2, 10, 10, 10), 'type_dict': {'pooling_data': np.float64}},
                {'ctx': mx.gpu(0), 'pooling_data': (2, 2, 10, 10, 10), 'type_dict': {'pooling_data': np.float32}},
                {'ctx': mx.gpu(0), 'pooling_data': (2, 2, 10, 10, 10), 'type_dict': {'pooling_data': np.float16}},
                {'ctx': mx.cpu(0), 'pooling_data': (2, 2, 10, 10, 10), 'type_dict': {'pooling_data': np.float64}},
                {'ctx': mx.cpu(0), 'pooling_data': (2, 2, 10, 10, 10), 'type_dict': {'pooling_data': np.float32}}]
    check_consistency(sym_3d, ctx_list_3d)

def test_regression_with_type()
    sym_logistic = mx.sym.LogisticRegressionOutput(name = 'regression')
    sym_linear = mx.sym.LinearRegressionOutput(name = 'regression')
    ctx_list = [{'ctx': mx.gpu(0), 'regression_data': (2, 2, 10, 10), 'type_dict': {'regression_data': np.float64}},
                {'ctx': mx.gpu(0), 'regression_data': (2, 2, 10, 10), 'type_dict': {'regression_data': np.float32}},
                {'ctx': mx.gpu(0), 'regression_data': (2, 2, 10, 10), 'type_dict': {'regression_data': np.float16}},
                {'ctx': mx.cpu(0), 'regression_data': (2, 2, 10, 10), 'type_dict': {'regression_data': np.float64}},
                {'ctx': mx.cpu(0), 'regression_data': (2, 2, 10, 10), 'type_dict': {'regression_data': np.float32}}]
    check_consistency(sym_logistic, ctx_list)
    check_consistency(sym_linear, ctx_list)

if __name__ == '__main__':
    test_batchnorm_with_type()
    test_convolution_with_type()
    test_deconvolution_with_type()
    test_upsampling_with_type()
    test_concat_with_type()
    test_elementwisesum_with_type()
    test_reshape_with_type()
    test_blockgrad_with_type()
    test_swapaxis_with_type()
    test_fullyconnected_with_type()
    test_activation_with_type()
    test_embedding_with_type()
    test_pooling_with_type()
    test_regression_with_type()
    #test_softmax_with_shape((3,4), mx.gpu())
    #test_multi_softmax_with_shape((3,4,5), mx.gpu())

