import sys
import os
curr_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
sys.path.insert(0, os.path.join(curr_path, '../unittest'))
from test_operator import *
from test_optimizer import *
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
    sym1 = mx.sym.Convolution(num_filter=3, kernel=(3,3), name='conv')

    data = mx.sym.Variable('conv_data')
    w = mx.sym.Variable('conv_weight')
    b = mx.sym.Variable('conv_bias')
    w = mx.sym.transpose(w, axes=(0,2,3,1))
    sym2 = mx.sym.transpose(data, axes=(0,2,3,1))
    sym2 = mx.sym.Convolution(sym2, w, b, layout='NHWC', num_filter=3, kernel=(3,3))
    sym2 = mx.sym.transpose(sym2, axes=(0,3,1,2), name='conv')

    sym = [sym1, sym1, sym1, sym1, sym1, sym2, sym2]
    ctx_list = [{'ctx': mx.gpu(0), 'conv_data': (2, 2, 10, 10), 'type_dict': {'conv_data': np.float64}},
                {'ctx': mx.gpu(0), 'conv_data': (2, 2, 10, 10), 'type_dict': {'conv_data': np.float32}},
                {'ctx': mx.gpu(0), 'conv_data': (2, 2, 10, 10), 'type_dict': {'conv_data': np.float16}},
                {'ctx': mx.cpu(0), 'conv_data': (2, 2, 10, 10), 'type_dict': {'conv_data': np.float64}},
                {'ctx': mx.cpu(0), 'conv_data': (2, 2, 10, 10), 'type_dict': {'conv_data': np.float32}},
                # NHWC
                {'ctx': mx.gpu(0), 'conv_data': (2, 2, 10, 10), 'conv_weight': (3, 2, 3, 3),
                 'type_dict': {'conv_data': np.float32, 'conv_weight': np.float32}},
                {'ctx': mx.gpu(0), 'conv_data': (2, 2, 10, 10), 'conv_weight': (3, 2, 3, 3),
                 'type_dict': {'conv_data': np.float16, 'conv_weight': np.float16}}
                ]
    check_consistency(sym, ctx_list)

def test_convolution_options():
    ctx_list = [{'ctx': mx.gpu(0), 'conv_data': (2, 2, 7, 7), 'type_dict': {'conv_data': np.float64}},
                {'ctx': mx.gpu(0), 'conv_data': (2, 2, 7, 7), 'type_dict': {'conv_data': np.float32}},
                {'ctx': mx.gpu(0), 'conv_data': (2, 2, 7, 7), 'type_dict': {'conv_data': np.float16}},
                {'ctx': mx.cpu(0), 'conv_data': (2, 2, 7, 7), 'type_dict': {'conv_data': np.float64}},
                {'ctx': mx.cpu(0), 'conv_data': (2, 2, 7, 7), 'type_dict': {'conv_data': np.float32}}]

    sym = mx.sym.Convolution(num_filter=3, kernel=(3,3), pad=(1,1), name='conv')
    check_consistency(sym, ctx_list)
    sym = mx.sym.Convolution(num_filter=3, kernel=(3,3), stride=(2,2), name='conv')
    check_consistency(sym, ctx_list)
    sym = mx.sym.Convolution(num_filter=3, kernel=(3,3), dilate=(2,2), name='conv')
    check_consistency(sym, ctx_list)

    ctx_list = [{'ctx': mx.gpu(0), 'conv_data': (2, 2, 5, 7, 7), 'type_dict': {'conv_data': np.float64}},
                {'ctx': mx.gpu(0), 'conv_data': (2, 2, 5, 7, 7), 'type_dict': {'conv_data': np.float32}}]
    sym = mx.sym.Convolution(num_filter=3, kernel=(2,3,3), pad=(1,1,1), name='conv')
    check_consistency(sym, ctx_list)
    sym = mx.sym.Convolution(num_filter=3, kernel=(2,3,3), stride=(2,2,2), name='conv')
    check_consistency(sym, ctx_list)

def test_pooling_with_type():
    ctx_list = [{'ctx': mx.gpu(0), 'pool_data': (2, 2, 10, 10), 'type_dict': {'pool_data': np.float64}},
                {'ctx': mx.gpu(0), 'pool_data': (2, 2, 10, 10), 'type_dict': {'pool_data': np.float32}},
                {'ctx': mx.gpu(0), 'pool_data': (2, 2, 10, 10), 'type_dict': {'pool_data': np.float16}},
                {'ctx': mx.cpu(0), 'pool_data': (2, 2, 10, 10), 'type_dict': {'pool_data': np.float64}},
                {'ctx': mx.cpu(0), 'pool_data': (2, 2, 10, 10), 'type_dict': {'pool_data': np.float32}}]
    sym = mx.sym.Pooling(kernel=(3,3), pool_type='max', pooling_convention='valid', name='pool')
    check_consistency(sym, ctx_list)

    sym = mx.sym.Pooling(kernel=(3,3), pool_type='max', pooling_convention='full', name='pool')
    check_consistency(sym, ctx_list)

    sym = mx.sym.Pooling(kernel=(300,300), pool_type='max', global_pool=True, name='pool')
    check_consistency(sym, ctx_list)

def test_deconvolution_with_type():
    sym = mx.sym.Deconvolution(num_filter=2, kernel=(3,3), name='deconv')
    ctx_list = [{'ctx': mx.gpu(0), 'deconv_data': (2, 2, 10, 10), 'type_dict': {'deconv_data': np.float64}},
                {'ctx': mx.gpu(0), 'deconv_data': (2, 2, 10, 10), 'type_dict': {'deconv_data': np.float32}},
                {'ctx': mx.gpu(0), 'deconv_data': (2, 2, 10, 10), 'type_dict': {'deconv_data': np.float16}},
                {'ctx': mx.cpu(0), 'deconv_data': (2, 2, 10, 10), 'type_dict': {'deconv_data': np.float64}},
                {'ctx': mx.cpu(0), 'deconv_data': (2, 2, 10, 10), 'type_dict': {'deconv_data': np.float32}}]
    check_consistency(sym, ctx_list)
    check_consistency(sym, ctx_list, grad_req="add")

def test_bilinear_sampler_with_type():
    data = mx.sym.Variable('data')
    grid = mx.sym.Variable('grid')
    sym = mx.sym.BilinearSampler(data=data, grid=grid)
    ctx_list = [{'ctx': mx.gpu(0), 'data': (1, 5, 10, 10), 'grid': (1, 2, 10, 10),
                 'type_dict': {'data': np.float64}},
                {'ctx': mx.gpu(0), 'data': (1, 5, 10, 10), 'grid': (1, 2, 10, 10),
                 'type_dict': {'data': np.float32}},
                {'ctx': mx.gpu(0), 'data': (1, 5, 10, 10), 'grid': (1, 2, 10, 10),
                 'type_dict': {'data': np.float16}},
                {'ctx': mx.cpu(0), 'data': (1, 5, 10, 10), 'grid': (1, 2, 10, 10),
                 'type_dict': {'data': np.float64}},
                {'ctx': mx.cpu(0), 'data': (1, 5, 10, 10), 'grid': (1, 2, 10, 10),
                 'type_dict': {'data': np.float32}}]
    check_consistency(sym, ctx_list)
    check_consistency(sym, ctx_list, grad_req="add")

def test_grid_generator_with_type():
    data = mx.sym.Variable('data')
    sym = mx.sym.GridGenerator(data=data, transform_type='affine', target_shape=(20, 20))
    ctx_list = [{'ctx': mx.gpu(0), 'data': (3, 6), 'type_dict': {'data': np.float32}},
                {'ctx': mx.cpu(0), 'data': (3, 6), 'type_dict': {'data': np.float32}}]
    check_consistency(sym, ctx_list)
    check_consistency(sym, ctx_list, grad_req="add")
    sym = mx.sym.GridGenerator(data=data, transform_type='warp', target_shape=(20, 20))
    ctx_list = [{'ctx': mx.gpu(0), 'data': (3, 2, 20, 20), 'type_dict': {'data': np.float32}},
                {'ctx': mx.cpu(0), 'data': (3, 2, 20, 20), 'type_dict': {'data': np.float32}}]
    check_consistency(sym, ctx_list)
    check_consistency(sym, ctx_list, grad_req="add")

def test_pooling_with_type():
    np.random.seed(1234)
    ctx_list = [{'ctx': mx.gpu(0), 'pool_data': (10, 2, 10, 10), 'type_dict': {'pool_data': np.float64}},
                {'ctx': mx.gpu(0), 'pool_data': (10, 2, 10, 10), 'type_dict': {'pool_data': np.float32}},
                {'ctx': mx.gpu(0), 'pool_data': (10, 2, 10, 10), 'type_dict': {'pool_data': np.float16}},
                {'ctx': mx.cpu(0), 'pool_data': (10, 2, 10, 10), 'type_dict': {'pool_data': np.float64}},
                {'ctx': mx.cpu(0), 'pool_data': (10, 2, 10, 10), 'type_dict': {'pool_data': np.float32}}]

    sym = mx.sym.Pooling(name='pool', kernel=(3,3), stride=(2,2), pool_type='max')
    check_consistency(sym, ctx_list)

    sym = mx.sym.Pooling(name='pool', kernel=(3,3), pad=(1,1), pool_type='avg')
    check_consistency(sym, ctx_list)

    # this is unstable
    # sym = mx.sym.Pooling(name='pool', kernel=(5,5), pad=(2,2), pool_type='max')
    # check_consistency(sym, ctx_list)

    sym = mx.sym.Pooling(name='pool', kernel=(3,3), pad=(1,1), pool_type='sum')
    check_consistency(sym, ctx_list)

def test_upsampling_with_type():
    sym = mx.sym.UpSampling(scale=2, num_filter=2, name='up', sample_type='nearest', num_args=1)
    ctx_list = [{'ctx': mx.gpu(0), 'up_arg0': (2, 2, 2, 10), 'type_dict': {'up_arg0': np.float64}},
                {'ctx': mx.gpu(0), 'up_arg0': (2, 2, 2, 10), 'type_dict': {'up_arg0': np.float32}},
                {'ctx': mx.gpu(0), 'up_arg0': (2, 2, 2, 10), 'type_dict': {'up_arg0': np.float16}},
                {'ctx': mx.cpu(0), 'up_arg0': (2, 2, 2, 10), 'type_dict': {'up_arg0': np.float64}},
                {'ctx': mx.cpu(0), 'up_arg0': (2, 2, 2, 10), 'type_dict': {'up_arg0': np.float32}}]
    check_consistency(sym, ctx_list)

def test_upsampling_bilinear_with_type():
    sym = mx.sym.UpSampling(scale=2, num_filter=2, name='up', sample_type='bilinear', num_args=1)
    ctx_list = [{'ctx': mx.gpu(0), 'up_data': (2, 2, 2, 10), 'type_dict': {'up_data': np.float64}},
                {'ctx': mx.gpu(0), 'up_data': (2, 2, 2, 10), 'type_dict': {'up_data': np.float32}},
                {'ctx': mx.gpu(0), 'up_data': (2, 2, 2, 10), 'type_dict': {'up_data': np.float16}},
                {'ctx': mx.cpu(0), 'up_data': (2, 2, 2, 10), 'type_dict': {'up_data': np.float64}},
                {'ctx': mx.cpu(0), 'up_data': (2, 2, 2, 10), 'type_dict': {'up_data': np.float32}}]
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
    arg_params = {'embedding_data': np.random.randint(low=0, high=10, size=(2, 10))}
    check_consistency(sym, ctx_list, grad_req={'embedding_data': 'null','embedding_weight': 'write'},
                      arg_params=arg_params)

def test_take_with_type():
    sym = mx.sym.take(name='take')
    for data_ndim in range(2, 5):
        for idx_ndim in range(1, 4):
            data_shape = ()
            for _ in range(data_ndim):
                data_shape += (np.random.randint(low=3, high=6), )
            idx_shape = ()
            for _ in range(idx_ndim):
                idx_shape += (np.random.randint(low=3, high=5), ) 
            ctx_list = [{'ctx': mx.gpu(0), 'take_indices': idx_shape, 
                         'take_a': data_shape, 
                         'type_dict': {'take_indices': np.float64, 
                                       'take_a': np.float64}},
                        {'ctx': mx.gpu(0), 'take_indices': idx_shape, 
                         'take_a': data_shape, 
                         'type_dict': {'take_indices': np.float32, 
                                       'take_a': np.float32}},
                        {'ctx': mx.gpu(0), 'take_indices': idx_shape, 
                         'take_a': data_shape, 
                         'type_dict': {'take_indices': np.float16, 
                                       'take_a': np.float16}},
                        {'ctx': mx.cpu(0), 'take_indices': idx_shape, 
                         'take_a': data_shape, 
                         'type_dict': {'take_indices': np.float64, 
                                       'take_a': np.float64}},
                        {'ctx': mx.cpu(0), 'take_indices': idx_shape, 
                         'take_a': data_shape, 
                         'type_dict': {'take_indices': np.float32, 
                                       'take_a': np.float32}},
                        {'ctx': mx.cpu(0), 'take_indices': idx_shape, 
                         'take_a': data_shape, 
                         'type_dict': {'take_indices': np.float16, 
                                       'take_a': np.float16}}]
            arg_params = {'take_indices': np.random.randint(low=0, 
                                                            high=data_shape[0], 
                                                            size=idx_shape), 
                          'take_a': np.random.normal(size=data_shape)}
            check_consistency(sym, ctx_list, 
                              grad_req={'take_indices': 'null',
                                        'take_a': 'write'},
                              arg_params=arg_params)

def check_rnn_consistency(cell1, cell2):
    dshape = (32, 5, 200)
    data = mx.sym.Variable('data')

    sym1, _ = cell1.unroll(5, data, merge_outputs=True)
    mod1 = mx.mod.Module(sym1, label_names=None, context=mx.gpu(0))
    mod1.bind(data_shapes=[('data', dshape)], label_shapes=None)

    sym2, _ = cell2.unroll(5, data, merge_outputs=True)
    mod2 = mx.mod.Module(sym2, label_names=None, context=mx.gpu(0))
    mod2.bind(data_shapes=[('data', dshape)], label_shapes=None)

    mod1.init_params()
    args, auxs = mod1.get_params()
    args = cell1.unpack_weights(args)
    args = cell2.pack_weights(args)
    mod2.set_params(args, auxs)

    batch=mx.io.DataBatch(data=[mx.random.uniform(shape=dshape)], label=[])
    mod1.forward(batch)
    mod2.forward(batch)

    assert_allclose(mod1.get_outputs()[0].asnumpy(), mod2.get_outputs()[0].asnumpy(), rtol=1e-2, atol=1e-4)


def test_rnn():
    fused = mx.rnn.FusedRNNCell(100, num_layers=2, mode='rnn_relu', prefix='')

    stack = mx.rnn.SequentialRNNCell()
    stack.add(mx.rnn.RNNCell(100, activation='relu', prefix='l0_'))
    stack.add(mx.rnn.RNNCell(100, activation='relu', prefix='l1_'))

    check_rnn_consistency(fused, stack)
    check_rnn_consistency(stack, fused)


def test_lstm():
    fused = mx.rnn.FusedRNNCell(100, num_layers=2, mode='lstm', prefix='')

    stack = mx.rnn.SequentialRNNCell()
    stack.add(mx.rnn.LSTMCell(100, prefix='l0_'))
    stack.add(mx.rnn.LSTMCell(100, prefix='l1_'))

    check_rnn_consistency(fused, stack)
    check_rnn_consistency(stack, fused)


if __name__ == '__main__':
    test_lstm()
    test_rnn()
    test_convolution_options()
    test_convolution_with_type()
    test_pooling_with_type()
    test_batchnorm_with_type()
    test_batchnorm_with_type()
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
    test_take_with_type()
    test_bilinear_sampler_with_type()
    test_grid_generator_with_type()
