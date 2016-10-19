import sys
import os
curr_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
sys.path.insert(0, os.path.join(curr_path, '../unittest'))
from test_operator import *
import mxnet as mx
import numpy as np
from mxnet.test_utils import check_consistency, set_default_context
from numpy.testing import assert_allclose
import time

set_default_context(mx.gpu(0))
del test_support_vector_machine_l1_svm
del test_support_vector_machine_l2_svm

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
    #test_softmax_with_shape((3,4), mx.gpu())
    #test_multi_softmax_with_shape((3,4,5), mx.gpu())

