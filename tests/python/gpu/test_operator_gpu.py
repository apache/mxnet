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

import sys
import os
import time
import multiprocessing as mp
import mxnet as mx
import numpy as np
import pytest
import itertools
import scipy.sparse as sps
import mxnet.ndarray.sparse as mxsps
from mxnet.test_utils import check_consistency, set_default_context, assert_almost_equal, assert_allclose
from mxnet.test_utils import check_symbolic_forward, check_symbolic_backward, discard_stderr
from mxnet.test_utils import default_context, rand_shape_2d, rand_ndarray, same, environment
from mxnet.base import MXNetError
from mxnet import autograd

curr_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
sys.path.insert(0, os.path.join(curr_path, '../unittest'))
from common import assert_raises_cudnn_not_satisfied, assert_raises_cuda_not_satisfied
from common import run_in_spawned_process
from test_operator import check_sequence_reverse, allclose_function
from test_operator import *
from test_numpy_ndarray import *
from test_numpy_op import *
from test_numpy_interoperability import *
from test_gluon_probability_v2 import *
from test_optimizer import *
from test_random import *
from test_exc_handling import *
from test_sparse_ndarray import *
from test_sparse_operator import *
from test_ndarray import *
from test_subgraph_op import *
from test_gluon_gpu import _test_bulking
from test_contrib_operator import test_multibox_target_op
from test_optimizer import test_adamW
del test_custom_op_fork  #noqa

set_default_context(mx.gpu(0))

def check_countsketch(in_dim,out_dim,n):
    data = mx.sym.Variable("data")
    h = mx.sym.Variable("h")
    s = mx.sym.Variable("s")
    sym = mx.sym.contrib.count_sketch(data=data, h=h, s=s, name='countsketch',out_dim = out_dim)
    shape = [(n,in_dim), (1,in_dim),(1,in_dim)]     #shape of input x, hash h and hash s

    arr = [mx.nd.empty(shape[i]) for i in range(3)]
    arr_grad = [mx.nd.empty(shape[i]) for i in range(3)]
    x = np.random.uniform(-10, 10, shape[0])
    arr[0][:] = x                                 #input x
    h = np.random.randint(0, out_dim, shape[1])
    arr[1][:] = h                                 #hash h
    s = np.random.randint(0, 2, shape[2])*2-np.ones(shape[2])
    arr[2][:] = s                                 #hash s
    locations = {"data": x, "h": h, "s": s}
    a = np.zeros((n,out_dim))
    temp = np.multiply(x, s)
    for num_sample in np.arange(0,n):
        for idx in np.arange(0,in_dim):
            a[num_sample][h[0][idx]] += temp[num_sample][idx]
    check_symbolic_forward(sym, locations, [a], rtol=1e-3, atol=1e-5, ctx=mx.gpu(0))
    out_grad = mx.nd.empty((n,out_dim))
    out_grad[:] = np.random.normal(-3, 3, (n,out_dim))
    a = np.zeros((n,in_dim))
    for j in np.arange(0,n):
        for i in np.arange(0,in_dim):
            a[j,i] = out_grad.asnumpy()[j, h[0,i]] * s[0,i]
    check_symbolic_backward(sym, locations, [out_grad], [a], rtol=1e-3, atol=1e-5, ctx=mx.gpu(0))


@pytest.mark.serial
def test_countsketch():
    minindim = 40
    maxindim = 100
    minoutdim = 5
    maxoutdim = 30
    maxn = 200
    in_dim = np.random.randint(minindim, maxindim)
    out_dim = np.random.randint(minoutdim, maxoutdim)
    n = np.random.randint(1, maxn)
    check_countsketch(in_dim, out_dim, n)


def check_fft(shape):
    sym = mx.sym.contrib.fft(name='fft', compute_size = 128)
    if len(shape) == 2:
        if shape[1]%2 != 0:
            lst = list(shape)
            lst[1] = lst[1]*2
            shape = tuple(lst)
            shape_old = shape
    if len(shape) == 4:
        if shape[3]%2 != 0:
            lst = list(shape)
            lst[3] = lst[3]*2
            shape = tuple(lst)
            shape_old = shape
    init = [np.random.normal(size=shape, scale=1.0)]
    arr_grad = [mx.nd.empty(shape)]
    ctx_list = [{'ctx': mx.gpu(0),'fft_data': shape, 'type_dict': {'fft_data': np.float32}}]
    exe_list = [sym._simple_bind(**ctx) for ctx in ctx_list]

    for exe in exe_list:
        for arr, iarr in zip(exe.arg_arrays, init):
            arr[:] = iarr.astype(arr.dtype)
    # forward
    for exe in exe_list:
        exe.forward(is_train=True)
    out1 = [exe.outputs[0].asnumpy() for exe in exe_list]
    out = np.fft.fft(init, n=None, axis=-1, norm=None)
    if len(shape) == 2:
        out = np.reshape(out,(out.shape[1],out.shape[2]))
        out2 = np.append(out.real, out.imag, axis = 1)
        a = np.zeros(out1[0].shape)
        p = 0
        for i in range(out2.shape[1]//2):
            a[:,p] = out2[:,i]
            a[:,p+1] = out2[:,i+out2.shape[1]//2]
            p = p+2

    if len(shape) == 4:
        out = np.reshape(out,(out.shape[1],out.shape[2],out.shape[3],out.shape[4]))
        out2 = np.append(out.real, out.imag, axis = 1)
        a = np.zeros(out1[0].shape)
        for i in range(out1[0].shape[0]):
            for j in range(out1[0].shape[1]):
                p = 0
                for k in range(out2.shape[3]):
                    a[i,j,:,p] = out2[i,j,:,k]
                    a[i,j,:,p+1] = out2[i,j+out1[0].shape[1],:,k]
                    p = p+2

    assert_almost_equal(a, out1[0], rtol=1e-3, atol=1e-5)

    # backward
    if len(shape) == 2:
        out_grad = mx.nd.empty((shape[0],2*shape[1]))
        out_grad[:] = np.random.normal(-3, 3, (shape[0],2*shape[1]))
        # out_grad_to_complex
        out_grad_complex = np.zeros(shape,dtype = np.complex64)
        for i in range(0,shape[1]):
            out_grad_complex.real[:,i] = out_grad.asnumpy()[:,2*i]
            out_grad_complex.imag[:,i] = out_grad.asnumpy()[:,2*i+1]
        for exe in exe_list:
            exe.backward([out_grad])
        a = np.fft.ifft(out_grad_complex, n=None, axis=-1, norm=None)
        assert_almost_equal(a.real, exe.grad_arrays[0]/shape[1],rtol=1e-3, atol=1e-5)

    if len(shape) == 4:
        out_grad = mx.nd.empty(out1[0].shape)
        out_grad[:] = np.random.normal(-3, 3, out1[0].shape)
        # out_grad_to_complex
        out_grad_complex = np.zeros(shape,dtype = np.complex64)
        for i in range(0,shape[3]):
            out_grad_complex.real[:,:,:,i] = out_grad.asnumpy()[:,:,:,2*i]
            out_grad_complex.imag[:,:,:,i] = out_grad.asnumpy()[:,:,:,2*i+1]
        for exe in exe_list:
            exe.backward([out_grad])
        a = np.fft.ifft(out_grad_complex, n=None, axis=-1, norm=None)
        assert_almost_equal(a.real, exe.grad_arrays[0]/shape[3],rtol=1e-3, atol=1e-5)

def test_fft():
    nrepeat = 2
    maxdim = 10
    for repeat in range(nrepeat):
        for order in [2,4]:
            shape = tuple(np.random.randint(1, maxdim, size=order))
            check_fft(shape)

def _make_ndarrays(input_list, ctx=mx.gpu(0)):
    return [mx.nd.array(arr, dtype=arr.dtype, ctx=ctx) for arr in input_list]

def check_multi_sum_sq(dtype, shapes, ctx, tol1, tol2):
    values_arr = [np.random.rand(*shape).astype(dtype) * 10. for shape in shapes]
    mx_vals = _make_ndarrays(values_arr, ctx=ctx)
    sum_sq = mx.nd.multi_sum_sq(*mx_vals, num_arrays=len(shapes))
    sum_sq2 = mx.nd.multi_sum_sq(*mx_vals, num_arrays=len(shapes))
    # checks that operator is deterministic
    assert np.array_equal(sum_sq.asnumpy(), sum_sq2.asnumpy())

    ref_sum_sq = mx.nd.array([(v.astype('float32') ** 2).sum() for v in values_arr],
                             dtype='float32', ctx=ctx)
    assert_almost_equal(ref_sum_sq.asnumpy(), sum_sq.asnumpy(), atol=tol1, rtol=tol1)

@pytest.mark.serial
def test_multi_sum_sq():
    min_nparam = 100
    max_nparam = 120
    min_dim = 50000
    max_dim = 100000
    max_ndim = 1

    dtypes = ['float16','float32', 'float64']
    for ctx in [mx.gpu(0)]:
        for dtype in dtypes:
            nparam = np.random.randint(min_nparam + 1, max_nparam + 1)
            shapes = [np.random.randint(min_dim, max_dim + 1, size=max_ndim) for i in range(nparam)]
            low_tol = ctx == mx.cpu(0) and ('float16'in [dtype])
            tol1 = 1e-3 if low_tol else 1e-5
            tol2 = 1e-6 if low_tol else 1e-7
            check_multi_sum_sq(dtype, shapes, ctx, tol1, tol2)

def check_fast_lars(w_dtype, g_dtype, shapes, ctx, tol1, tol2):
    weights_arr = [np.random.rand(*shape).astype(w_dtype) * 10. for shape in shapes]
    grads_arr = [np.random.rand(*shape).astype(g_dtype) for shape in shapes]

    lrs = (np.random.rand(len(shapes)).astype('float32') + 0.1) / 100.
    wds = (np.random.rand(len(shapes)).astype('float32') + 0.1) / 1000.
    eta = (np.random.rand() + 0.1)
    eps = (np.random.rand() + 0.1) / 10000.

    mx_w = _make_ndarrays(weights_arr, ctx=ctx)
    mx_g = _make_ndarrays(grads_arr, ctx=ctx)
    mx_lrs = mx.nd.array(lrs, dtype='float32', ctx=ctx)
    mx_wds = mx.nd.array(wds, dtype='float32', ctx=ctx)

    w_sum_sq = mx.nd.multi_sum_sq(*mx_w, num_arrays=len(shapes))
    g_sum_sq = mx.nd.multi_sum_sq(*mx_g, num_arrays=len(shapes))

    ref_w_sum_sq = mx.nd.array([(w.astype('float32') ** 2).sum() for w in weights_arr],
                                dtype='float32', ctx=ctx)
    ref_g_sum_sq = mx.nd.array([(g.astype('float32') ** 2).sum() for g in grads_arr],
                                dtype='float32', ctx=ctx)
    assert_almost_equal(ref_w_sum_sq.asnumpy(), w_sum_sq.asnumpy(), atol=tol1, rtol=tol1)
    assert_almost_equal(ref_g_sum_sq.asnumpy(), g_sum_sq.asnumpy(), atol=tol1, rtol=tol1)

    rescale_grad = (np.random.rand() + 0.5) * 100.
    mx_new_lrs = mx.nd.multi_lars(mx_lrs, w_sum_sq, g_sum_sq, mx_wds, eta=eta, eps=eps,
                                  rescale_grad=rescale_grad)
    ref_w_l2norm = mx.nd.sqrt(ref_w_sum_sq)
    ref_g_l2norm = mx.nd.sqrt(ref_g_sum_sq * rescale_grad * rescale_grad)
    ref_new_lrs = mx.nd.zeros(ref_w_l2norm.shape, dtype='float32', ctx=ctx)
    for i in range(ref_w_l2norm.size):
        _w = ref_w_l2norm[i]
        _g = ref_g_l2norm[i]
        if _w > 0.0 and _g > 0.0:
            ref_new_lrs[i] = lrs[i] * eta * _w / (_g + wds[i] * _w + eps)
        else:
            ref_new_lrs[i] = lrs[i]
    assert_almost_equal(ref_new_lrs.asnumpy(), mx_new_lrs.asnumpy(), atol=tol2, rtol=tol2)

@pytest.mark.serial
def test_fast_lars():
    min_nparam = 50
    max_nparam = 60
    maxdim = 10000
    maxndim = 1

    dtypes = ['float16','float32', 'float64']
    for ctx in [mx.cpu(0), mx.gpu(0)]:
        for w_dtype in dtypes:
            for g_dtype in dtypes:
                nparam = np.random.randint(min_nparam + 1, max_nparam + 1)
                shapes = [np.random.randint(1, maxdim + 1, size=maxndim) for i in range(nparam)]
                lowTol = ctx == mx.cpu(0) and ('float16'in [w_dtype, g_dtype])
                tol1 = 1e-3 if lowTol else 1e-5
                tol2 = 1e-6 if lowTol else 1e-7
                check_fast_lars(w_dtype, g_dtype, shapes, ctx, tol1, tol2)

def check_preloaded_multi_sgd(dtype, shapes, momentum, use_master_weights):
    def _flatten_list(nested_list):
        return [item for sublist in nested_list for item in sublist]
    weights_arr = [np.random.rand(*shape).astype(dtype) * 100. for shape in shapes]
    grads_arr = [np.random.rand(*shape).astype(dtype) * 100. for shape in shapes]
    rescale_grad = (np.random.random() + 1.0)
    mx_w = _make_ndarrays(weights_arr)
    mx_g = _make_ndarrays(grads_arr)
    mx_p_w = _make_ndarrays(weights_arr)
    mx_p_g = _make_ndarrays(grads_arr)
    lrs = list((np.random.random(size=len(shapes)).astype('float32') + 0.1) / 100.)
    mx_lrs = mx.nd.array(lrs, dtype='float32', ctx=mx.gpu(0))
    wds = list((np.random.random(size=len(shapes)).astype('float32') + 0.1) / 1000.)
    mx_wds = mx.nd.array(wds, dtype='float32', ctx=mx.gpu(0))
    if use_master_weights:
        weights32_arr = [arr.astype('float32') for arr in weights_arr]
        mx_w32 = _make_ndarrays(weights32_arr)
        mx_p_w32 = _make_ndarrays(weights32_arr)
    if momentum is None:
        if use_master_weights:
            mx.nd.multi_mp_sgd_update(
                                     *_flatten_list(zip(mx_w, mx_g, mx_w32)),
                                     num_weights=len(shapes), lrs=lrs, wds=wds,
                                     rescale_grad=rescale_grad, out=mx_w)
            mx.nd.preloaded_multi_mp_sgd_update(
                                     *(_flatten_list(zip(mx_p_w, mx_p_g, mx_p_w32)) +
                                      [mx_lrs, mx_wds]), num_weights=len(shapes),
                                     rescale_grad=rescale_grad, out=mx_p_w)
        else:
            out = mx.nd.multi_sgd_update(
                                    *_flatten_list(zip(mx_w, mx_g)),
                                    num_weights=len(shapes), lrs=lrs, wds=wds,
                                    rescale_grad=rescale_grad, out=mx_w)
            preloaded_out = mx.nd.preloaded_multi_sgd_update(
                                    *(_flatten_list(zip(mx_p_w, mx_p_g)) +
                                     [mx_lrs, mx_wds]), num_weights=len(shapes),
                                    rescale_grad=rescale_grad, out=mx_p_w)
    else:
        if use_master_weights:
            momentums_arr = [np.random.rand(*shape).astype("float32") for shape in shapes]
            mx_m = _make_ndarrays(momentums_arr)
            mx_p_m = _make_ndarrays(momentums_arr)
            out = mx.nd.multi_mp_sgd_mom_update(
                                    *_flatten_list(zip(mx_w, mx_g, mx_m, mx_w32)),
                                    num_weights=len(shapes), lrs=lrs, wds=wds,
                                    rescale_grad=0.95, momentum=momentum, out=mx_w)
            preloaded_out = mx.nd.preloaded_multi_mp_sgd_mom_update(
                                    *(_flatten_list(zip(mx_p_w, mx_p_g, mx_p_m, mx_p_w32)) +
                                      [mx_lrs, mx_wds]), num_weights=len(shapes),
                                    rescale_grad=0.95, momentum=momentum, out=mx_p_w)
        else:
            momentums_arr = [np.random.rand(*shape).astype(dtype) for shape in shapes]
            mx_m = _make_ndarrays(momentums_arr)
            mx_p_m = _make_ndarrays(momentums_arr)
            mx.nd.multi_sgd_mom_update(
                                    *_flatten_list(zip(mx_w, mx_g, mx_m)),
                                    num_weights=len(shapes), lrs=lrs, wds=wds,
                                    rescale_grad=0.95, momentum=momentum, out=mx_w)
            mx.nd.preloaded_multi_sgd_mom_update(
                                    *(_flatten_list(zip(mx_p_w, mx_p_g, mx_p_m)) +
                                      [mx_lrs, mx_wds]), num_weights=len(shapes),
                                    rescale_grad=0.95, momentum=momentum, out=mx_p_w)

    def _assert_all_almost_equal(lhs_list, rhs_list, rtol, atol):
        for i, (lhs, rhs) in enumerate(zip(lhs_list, rhs_list)):
            assert_almost_equal(lhs.asnumpy(), rhs.asnumpy(), rtol=rtol, atol=atol)
    if dtype == 'float16':
        rtol = 1e-3
        atol = 1e-2
    else:
        rtol = 1e-5
        atol = 1e-6
    _assert_all_almost_equal(mx_p_w, mx_w, rtol, atol)
    if momentum is not None:
        _assert_all_almost_equal(mx_p_m, mx_m, rtol, atol)
    if use_master_weights:
        _assert_all_almost_equal(mx_p_w32, mx_w32, 1e-5, 1e-6)

def test_preloaded_multi_sgd():
    dtypes = ['float16', 'float32']
    momentums = [None, 0.9]
    min_nparam = 5
    max_nparam = 10
    maxdim = 6
    maxndim = 4
    for dtype in dtypes:
        use_master_weights_list = [False,] if dtype == 'float32' else [True, False]
        for use_master_weights in use_master_weights_list:
            for momentum in momentums:
                nparam = np.random.randint(min_nparam + 1, max_nparam + 1)
                shapes = [np.random.randint(1, maxdim + 1, size=maxndim) for i in range(nparam)]
                check_preloaded_multi_sgd(dtype, shapes, momentum, use_master_weights)


@pytest.mark.serial
@pytest.mark.flaky
def test_batchnorm_with_type():
  ctx_list_v2_2D = [
    {'ctx': mx.cpu(0), 'norm_data': (5, 2, 5, 5), 'type_dict': {'norm_data': np.float32}},
    {'ctx': mx.cpu(0), 'norm_data': (5, 2, 5, 5), 'type_dict': {'norm_data': np.float16}},
    {'ctx': mx.cpu(0), 'norm_data': (5, 2, 5, 5), 'type_dict': {'norm_data': np.float64}},
    {'ctx': mx.gpu(0), 'norm_data': (5, 2, 5, 5), 'type_dict': {'norm_data': np.float32}},
    {'ctx': mx.gpu(0), 'norm_data': (5, 2, 5, 5), 'type_dict': {'norm_data': np.float16}},
    {'ctx': mx.gpu(0), 'norm_data': (5, 2, 5, 5), 'type_dict': {'norm_data': np.float64}},
  ]

  ctx_list_v2_1D = [
    {'ctx': mx.cpu(0), 'norm_data': (5, 2, 5), 'type_dict': {'norm_data': np.float16}},
    {'ctx': mx.cpu(0), 'norm_data': (5, 2, 5), 'type_dict': {'norm_data': np.float32}},
    {'ctx': mx.cpu(0), 'norm_data': (5, 2, 5), 'type_dict': {'norm_data': np.float64}},
    {'ctx': mx.gpu(0), 'norm_data': (5, 2, 5), 'type_dict': {'norm_data': np.float16}},
    {'ctx': mx.gpu(0), 'norm_data': (5, 2, 5), 'type_dict': {'norm_data': np.float32}},
    {'ctx': mx.gpu(0), 'norm_data': (5, 2, 5), 'type_dict': {'norm_data': np.float64}},
  ]

  ctx_list_v2_3D = [
    {'ctx': mx.cpu(0), 'norm_data': (3, 2, 3, 2, 3), 'type_dict': {'norm_data': np.float16}},
    {'ctx': mx.cpu(0), 'norm_data': (3, 2, 3, 2, 3), 'type_dict': {'norm_data': np.float32}},
    {'ctx': mx.cpu(0), 'norm_data': (3, 2, 3, 2, 3), 'type_dict': {'norm_data': np.float64}},
    {'ctx': mx.gpu(0), 'norm_data': (3, 2, 3, 2, 3), 'type_dict': {'norm_data': np.float16}},
    {'ctx': mx.gpu(0), 'norm_data': (3, 2, 3, 2, 3), 'type_dict': {'norm_data': np.float32}},
    {'ctx': mx.gpu(0), 'norm_data': (3, 2, 3, 2, 3), 'type_dict': {'norm_data': np.float64}}
  ]

  # V2, 2D
  bools = [False, True]
  for fix_gamma, cudnn_off in itertools.product(bools, bools):
      sym = mx.sym.BatchNorm(name='norm', fix_gamma=fix_gamma, cudnn_off=cudnn_off)
      check_consistency(sym, ctx_list_v2_2D)

  # V2, 1D
  for fix_gamma, cudnn_off in itertools.product(bools, bools):
      sym = mx.sym.BatchNorm(name='norm', fix_gamma=fix_gamma, cudnn_off=cudnn_off)
      check_consistency(sym, ctx_list_v2_1D)

  # V2, 3D
  for fix_gamma, cudnn_off in itertools.product(bools, [True,]):
      sym = mx.sym.BatchNorm(name='norm', fix_gamma=fix_gamma, cudnn_off=cudnn_off)
      check_consistency(sym, ctx_list_v2_3D)


@pytest.mark.serial
def test_batchnorm_versions():
  def test_batchnorm_versions_helper(batchnorm_op_list, data, fix_gamma, use_global_stats):
    ctx_list = []
    sym_list = []

    # BatchNorm cpu
    if 'batchnorm_cpu' in batchnorm_op_list:
      ctx_list.append({'ctx': mx.cpu(0), 'batchnorm_data': data, 'type_dict': {'batchnorm_data': np.float32}})
      sym_list.append(mx.sym.BatchNorm(fix_gamma=fix_gamma,
                                       use_global_stats=use_global_stats,
                                       name='batchnorm'))

    # BatchNorm gpu (organic)
    if 'batchnorm_gpu' in batchnorm_op_list:
      ctx_list.append({'ctx': mx.gpu(0), 'batchnorm_data': data, 'type_dict': {'batchnorm_data': np.float32}})
      sym_list.append(mx.sym.BatchNorm(fix_gamma=fix_gamma,
                                       use_global_stats=use_global_stats,
                                       name='batchnorm', cudnn_off=True))

    # BatchNorm gpu cudnn (if cudnn is enabled)
    if 'batchnorm_cudnn' in batchnorm_op_list:
      ctx_list.append({'ctx': mx.gpu(0), 'batchnorm_data': data, 'type_dict': {'batchnorm_data': np.float32}})
      sym_list.append(mx.sym.BatchNorm(fix_gamma=fix_gamma,
                                       use_global_stats=use_global_stats,
                                       name='batchnorm', cudnn_off=False))

    check_consistency(sym_list, ctx_list)

  def test_1d_batchnorm(fix_gamma, use_global_stats):
    data = (2, 3, 20)
    test_batchnorm_versions_helper(batchnorm_op_list=['batchnorm_cpu',
                                                      'batchnorm_gpu', 'batchnorm_cudnn'],
                                   data=data,
                                   fix_gamma=fix_gamma, use_global_stats=use_global_stats)

  def test_2d_batchnorm(fix_gamma, use_global_stats):
    data = (2, 3, 10, 10)
    test_batchnorm_versions_helper(batchnorm_op_list=['batchnorm_cpu',
                                                      'batchnorm_gpu', 'batchnorm_cudnn'],
                                   data=data,
                                   fix_gamma=fix_gamma, use_global_stats=use_global_stats)

  def test_3d_batchnorm(fix_gamma, use_global_stats):
    data = (2, 3, 3, 5, 5)
    test_batchnorm_versions_helper(batchnorm_op_list=['batchnorm_cpu',
                                                      'batchnorm_gpu'],
                                   data=data,
                                   fix_gamma=fix_gamma, use_global_stats=use_global_stats)

  test_1d_batchnorm(True,  False)
  test_1d_batchnorm(False, False)
  test_1d_batchnorm(False, True)
  test_1d_batchnorm(True,  True)

  test_2d_batchnorm(True,  False)
  test_2d_batchnorm(False, False)
  test_2d_batchnorm(False, True)
  test_2d_batchnorm(True,  True)

  test_3d_batchnorm(True,  False)
  test_3d_batchnorm(False, False)
  test_3d_batchnorm(False, True)
  test_3d_batchnorm(True,  True)


@pytest.mark.seed(1234)
@assert_raises_cudnn_not_satisfied(min_version='5.1.10')
@pytest.mark.serial
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
    # wider tolerance needed for true-fp16 NCHW test above
    tol = {np.dtype(np.float16): 0.5,
               np.dtype(np.float32): 1e-3,
               np.dtype(np.float64): 1e-5,
               np.dtype(np.uint8): 0,
               np.dtype(np.int32): 0}
    check_consistency(sym, ctx_list, rtol=tol, atol=tol)
    # test ability to turn off training on bias
    check_consistency(sym, ctx_list, grad_req={'conv_data': 'write', 'conv_weight': 'write', 'conv_bias': 'null'}, rtol=tol, atol=tol)


# Apply N symbols against each of M contexts, checking that all NxM combinations match.
def check_consistency_NxM(sym_list, ctx_list):
    # e.g. if sym_list=[sym1, sym2] and ctx_list=[ctx1, ctx2, ctx3], then resulting lists are:
    # sym_list=[sym1, sym1, sym1, sym2, sym2, sym2] and ctx_list=[ctx1, ctx2, ctx3, ctx1, ctx2, ctx3]
    check_consistency(np.repeat(sym_list, len(ctx_list)), ctx_list * len(sym_list), scale=0.5)


@pytest.mark.skip(reason="test fails intermittently. temporarily disabled till it gets fixed. tracked at https://github.com/apache/incubator-mxnet/issues/10141")
@pytest.mark.serial
def test_convolution_options():
    # 1D convolution
    ctx_list = [{'ctx': mx.gpu(0), 'conv_data': (2, 2, 7), 'type_dict': {'conv_data': np.float64}},
                {'ctx': mx.gpu(0), 'conv_data': (2, 2, 7), 'type_dict': {'conv_data': np.float32}},
                {'ctx': mx.gpu(0), 'conv_data': (2, 2, 7), 'type_dict': {'conv_data': np.float16}},
                {'ctx': mx.cpu(0), 'conv_data': (2, 2, 7), 'type_dict': {'conv_data': np.float64}},
                {'ctx': mx.cpu(0), 'conv_data': (2, 2, 7), 'type_dict': {'conv_data': np.float32}}]
    # Pad > 0
    sym = mx.sym.Convolution(layout='NCW', num_filter=3, kernel=(3,), pad=(1,), name='conv')
    sym_no_cudnn = mx.sym.Convolution(num_filter=3, kernel=(3,), pad=(1,), cudnn_off=True, name='conv')
    check_consistency_NxM([sym, sym_no_cudnn], ctx_list)
    # Stride > 1
    sym = mx.sym.Convolution(layout='NCW', num_filter=3, kernel=(3,), stride=(2,), name='conv')
    sym_no_cudnn = mx.sym.Convolution(num_filter=3, kernel=(3,), stride=(2,), cudnn_off=True, name='conv')
    check_consistency_NxM([sym, sym_no_cudnn], ctx_list)
    # Dilate > 1
    sym = mx.sym.Convolution(layout='NCW', num_filter=3, kernel=(3,), dilate=(2,), name='conv')
    sym_no_cudnn = mx.sym.Convolution(num_filter=3, kernel=(3,), dilate=(2,), cudnn_off=True, name='conv')
    check_consistency_NxM([sym, sym_no_cudnn], ctx_list)
    # 1x1 convolution
    sym = mx.sym.Convolution(layout='NCW', num_filter=3, kernel=(1,), pad=(0,), name='conv')
    sym_no_cudnn = mx.sym.Convolution(num_filter=3, kernel=(1,), pad=(0,), cudnn_off=True, name='conv')
    check_consistency_NxM([sym, sym_no_cudnn], ctx_list)

    # 2D convolution
    ctx_list = [{'ctx': mx.gpu(0), 'conv_data': (2, 2, 7, 7), 'type_dict': {'conv_data': np.float64}},
                {'ctx': mx.gpu(0), 'conv_data': (2, 2, 7, 7), 'type_dict': {'conv_data': np.float32}},
                {'ctx': mx.gpu(0), 'conv_data': (2, 2, 7, 7), 'type_dict': {'conv_data': np.float16}},
                {'ctx': mx.cpu(0), 'conv_data': (2, 2, 7, 7), 'type_dict': {'conv_data': np.float64}},
                {'ctx': mx.cpu(0), 'conv_data': (2, 2, 7, 7), 'type_dict': {'conv_data': np.float32}}]
    # Pad > 0
    sym = mx.sym.Convolution(num_filter=3, kernel=(3,3), pad=(1,1), name='conv')
    sym_no_cudnn = mx.sym.Convolution(num_filter=3, kernel=(3,3), pad=(1,1), cudnn_off=True, name='conv')
    check_consistency_NxM([sym, sym_no_cudnn], ctx_list)
    # Stride > 1
    sym = mx.sym.Convolution(num_filter=3, kernel=(3,3), stride=(2,2), name='conv')
    sym_no_cudnn = mx.sym.Convolution(num_filter=3, kernel=(3,3), stride=(2,2), cudnn_off=True, name='conv')
    check_consistency_NxM([sym, sym_no_cudnn], ctx_list)
    # Dilate > 1
    sym = mx.sym.Convolution(num_filter=3, kernel=(3,3), dilate=(2,2), name='conv')
    sym_no_cudnn = mx.sym.Convolution(num_filter=3, kernel=(3,3), dilate=(2,2), cudnn_off=True, name='conv')
    check_consistency_NxM([sym, sym_no_cudnn], ctx_list)
    # 1x1 convolution
    sym = mx.sym.Convolution(num_filter=3, kernel=(1,1), pad=(0,0), name='conv')
    sym_no_cudnn = mx.sym.Convolution(num_filter=3, kernel=(1,1), pad=(0,0), cudnn_off=True, name='conv')
    check_consistency_NxM([sym, sym_no_cudnn], ctx_list)

    # 3D convolution
    ctx_list = [{'ctx': mx.cpu(0), 'conv_data': (2, 2, 5, 7, 7), 'type_dict': {'conv_data': np.float64}},
                {'ctx': mx.cpu(0), 'conv_data': (2, 2, 5, 7, 7), 'type_dict': {'conv_data': np.float64}},
                {'ctx': mx.gpu(0), 'conv_data': (2, 2, 5, 7, 7), 'type_dict': {'conv_data': np.float64}},
                {'ctx': mx.gpu(0), 'conv_data': (2, 2, 5, 7, 7), 'type_dict': {'conv_data': np.float32}}]
    # Pad > 0
    sym = mx.sym.Convolution(num_filter=3, kernel=(2,3,3), pad=(1,1,1), name='conv')
    sym_no_cudnn = mx.sym.Convolution(num_filter=3, kernel=(2,3,3), pad=(1,1,1), cudnn_off=True, name='conv')
    check_consistency_NxM([sym, sym_no_cudnn], ctx_list)
    # Stride > 1
    sym = mx.sym.Convolution(num_filter=3, kernel=(2,3,3), stride=(2,2,2), name='conv')
    sym_no_cudnn = mx.sym.Convolution(num_filter=3, kernel=(2,3,3), stride=(2,2,2), cudnn_off=True, name='conv')
    check_consistency_NxM([sym, sym_no_cudnn], ctx_list)
    # 1x1 convolution
    sym = mx.sym.Convolution(num_filter=3, kernel=(1,1,1), pad=(0,0,0), name='conv')
    sym_no_cudnn = mx.sym.Convolution(num_filter=3, kernel=(1,1,1), pad=(0,0,0), cudnn_off=True, name='conv')
    check_consistency_NxM([sym, sym_no_cudnn], ctx_list)


@pytest.mark.serial
def test_conv_deconv_guards():
    # Test cases for convolution and deconvolution via strided fft.  Ensure that the framework
    # guards against problematic CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT_TILING in cuDNN [7.3.1,7.5)
    # see https://docs.nvidia.com/deeplearning/sdk/cudnn-release-notes/rel_750.html#rel_750
    for (op, opname) in [(mx.sym.Convolution, 'conv'), (mx.sym.Deconvolution, 'deconv')]:
        dataname = opname + '_data'
        ctx = {'ctx': mx.gpu(0), dataname: (32, 32, 64, 64), 'type_dict': {dataname: np.float32}}
        test_cases = [
            {'num_filter':32, 'kernel':(6,6), 'pad':(0,0), 'stride':(2,2), 'name': opname},
            {'num_filter':32, 'kernel':(6,6), 'pad':(1,1), 'stride':(2,2), 'name': opname},
            {'num_filter':32, 'kernel':(6,7), 'pad':(0,1), 'stride':(2,2), 'name': opname},
            {'num_filter':32, 'kernel':(7,6), 'pad':(1,0), 'stride':(2,2), 'name': opname},
            {'num_filter':32, 'kernel':(7,7), 'pad':(0,0), 'stride':(2,2), 'name': opname},
            {'num_filter':32, 'kernel':(7,7), 'pad':(1,1), 'stride':(2,2), 'name': opname}]
        for test_case_args in test_cases:
            try:
                sym = op(**test_case_args)
                sym_no_cudnn = op(cudnn_off=True, **test_case_args)
                check_consistency([sym, sym_no_cudnn], [ctx, ctx], scale=0.1)
            except:
                print('Test failure of mx.sym.{} with args: {}'.format(op.__name__, test_case_args))
                raise


def _conv_with_num_streams(seed):
    with random_seed(seed):
        # Try to expose timing-dependent improper workspace sharing by parallel dgrad and wgrad
        num_trials = 20
        for _ in range(num_trials):
            size = np.random.randint(32, 128)
            # The cudnn conv operator runs dgrad and wgrad in separate streams if enabled, with possible
            # kernel overlap.  The non-cudnn conv op doesn't do this so is used as the 'golden copy'.
            ctx = {'ctx': mx.gpu(0), 'conv_data': (2, 2, size, size),
                   'type_dict': {'conv_data': np.float32}}
            # Adding 'flip' here isolates the model from the input node (which can't use inplace store)
            flipped = mx.sym.flip(axis=0, name='conv')
            sym = mx.sym.Convolution(data=flipped, num_filter=3, kernel=(3,3), pad=(1,1), name='conv')
            flipped_no_cudnn = mx.sym.flip(axis=0, name='conv')
            sym_no_cudnn = mx.sym.Convolution(data=flipped_no_cudnn, num_filter=3, kernel=(3,3), pad=(1,1),
                                              cudnn_off=True, name='conv')
            try:
                # tol can be pretty high- we're looking for a large diff due to garbaged workspace
                check_consistency([sym, sym_no_cudnn], [ctx, ctx], rtol=1e-2, atol=1e-2)
            except:
                print('Failing conv size = {}'.format(size))
                raise


@pytest.mark.skip(reason="skipping for now due to severe flakiness")
def test_convolution_multiple_streams():
    for num_streams in ['1', '2']:
        for engine in ['NaiveEngine', 'ThreadedEngine', 'ThreadedEnginePerDevice']:
            print('Starting engine {} with {} streams.'.format(engine, num_streams), file=sys.stderr)
            run_in_spawned_process(_conv_with_num_streams,
                {'MXNET_GPU_WORKER_NSTREAMS' : num_streams, 'MXNET_ENGINE_TYPE' : engine})
            print('Finished engine {} with {} streams.'.format(engine, num_streams), file=sys.stderr)


# This test is designed to expose an issue with cudnn v7.1.4 algo find() when invoked with large c.
# Algos returned by find() can fail to run with grad_req='add' (wgrad kernel beta parameter == 1.0f).
@pytest.mark.serial
def test_convolution_large_c():
    problematic_c = 64 * 1024
    # The convolution accumulates many values, so scale the input magnitude.
    scale = 0.1
    def test_1D_with_width(width, grad_req):
        ctx_list = [{'ctx': mx.gpu(0), 'conv_data': (1, problematic_c, width), 'type_dict': {'conv_data': np.float32}},
                    {'ctx': mx.gpu(0), 'conv_data': (1, problematic_c, width), 'type_dict': {'conv_data': np.float64}}]
        sym = mx.sym.Convolution(layout='NCW', num_filter=8, kernel=(2,), name='conv')
        check_consistency([sym, sym], ctx_list, grad_req=grad_req, scale=scale)

    def test_2D_with_width(width, grad_req):
        ctx_list = [{'ctx': mx.gpu(0), 'conv_data': (1, problematic_c, 2, width), 'type_dict': {'conv_data': np.float32}},
                    {'ctx': mx.gpu(0), 'conv_data': (1, problematic_c, 2, width), 'type_dict': {'conv_data': np.float64}}]
        sym = mx.sym.Convolution(layout='NCHW', num_filter=4, kernel=(2,2), name='conv')
        check_consistency([sym, sym], ctx_list, grad_req=grad_req, scale=scale)

    # Run with different data tensor shapes to run cudnnFind() multiple times.
    # First, populate algo and op caches with models that always use cudnnFind() (req == 'write').
    # Then run models that must avoid cached cudnnFind() results in some cases (req == 'add').
    widths = [4, 16, 64]
    for req in ['write', 'add']:
        for width in widths:
            test_1D_with_width(width, req)
            test_2D_with_width(width, req)


# This test is designed to expose an issue with cudnn v7.1.4 algo find() when invoked with large c.
# Algos returned by find() can fail to run with grad_req='add' (wgrad kernel beta parameter == 1.0f).
@pytest.mark.serial
def test_deconvolution_large_c():
    problematic_c = 64 * 1024
    # The deconvolution accumulates many values, so scale the input magnitude.
    scale = 0.1
    def test_1D_with_width(width, grad_req):
        ctx_list = [{'ctx': mx.gpu(0), 'deconv_data': (1, 8, width), 'type_dict': {'deconv_data': np.float32}},
                    {'ctx': mx.gpu(0), 'deconv_data': (1, 8, width), 'type_dict': {'deconv_data': np.float64}}]
        sym = mx.sym.Deconvolution(layout='NCW', num_filter=problematic_c, kernel=(2,), name='deconv')
        check_consistency([sym, sym], ctx_list, grad_req=grad_req, scale=scale)

    def test_2D_with_width(width, grad_req):
        ctx_list = [{'ctx': mx.gpu(0), 'deconv_data': (1, 8, 2, width), 'type_dict': {'deconv_data': np.float32}},
                    {'ctx': mx.gpu(0), 'deconv_data': (1, 8, 2, width), 'type_dict': {'deconv_data': np.float64}}]
        sym = mx.sym.Deconvolution(layout='NCHW', num_filter=problematic_c, kernel=(2,2), name='deconv')
        check_consistency([sym, sym], ctx_list, grad_req=grad_req, scale=scale)

    # Run with different data tensor shapes to run cudnnFind() multiple times.
    # First, populate algo and op caches with models that always use cudnnFind() (req == 'write').
    # Then run models that must avoid cached cudnnFind() results in some cases (req == 'add').
    widths = [4, 16, 64]
    for req in ['write', 'add']:
        for width in widths:
            test_1D_with_width(width, req)
            test_2D_with_width(width, req)


@pytest.mark.serial
def test_convolution_versions():
    # 2D convolution NCHW
    ctx_list = [{'ctx': mx.cpu(0), 'conv_data': (2, 2, 7, 7), 'type_dict': {'conv_data': np.float32}},
                {'ctx': mx.cpu(0), 'conv_data': (2, 2, 7, 7), 'type_dict': {'conv_data': np.float32}},
                {'ctx': mx.gpu(0), 'conv_data': (2, 2, 7, 7), 'type_dict': {'conv_data': np.float32}}]
    conv_cudnn = mx.sym.Convolution(num_filter=3, kernel=(3,3), pad=(1,1), name='conv')
    conv_cpu = mx.sym.Convolution(num_filter=3, kernel=(3,3), pad=(1,1), name='conv')
    conv_gpu = mx.sym.Convolution(num_filter=3, kernel=(3,3), pad=(1,1), cudnn_off=True, name='conv')
    syms = [conv_cudnn, conv_cpu, conv_gpu]
    check_consistency(syms, ctx_list)

    # 3D convolution NCDHW
    ctx_list = [{'ctx': mx.gpu(0), 'conv_data': (2, 2, 5, 7, 7), 'type_dict': {'conv_data': np.float32}},
                {'ctx': mx.cpu(0), 'conv_data': (2, 2, 5, 7, 7), 'type_dict': {'conv_data': np.float32}},
                {'ctx': mx.gpu(0), 'conv_data': (2, 2, 5, 7, 7), 'type_dict': {'conv_data': np.float32}}]
    conv_cudnn = mx.sym.Convolution(num_filter=3, kernel=(2,3,3), pad=(1,1,1), name='conv')
    conv_cpu = mx.sym.Convolution(num_filter=3, kernel=(2,3,3), pad=(1,1,1), name='conv')
    conv_gpu = mx.sym.Convolution(num_filter=3, kernel=(2,3,3), pad=(1,1,1), cudnn_off=True, name='conv')
    syms = [conv_cudnn, conv_cpu, conv_gpu]
    check_consistency(syms, ctx_list)


# More max-pooling strides and pads to test cudnn pooling implementation code paths
@pytest.mark.serial
def test_pooling_nhwc_with_convention():
    def make_pooling_syms(**kwargs):
        # Conventional NCHW layout pooling
        sym = mx.sym.Pooling(**kwargs)
        # NHWC pooling
        data = mx.sym.Variable('pool_data')
        sym_nhwc = mx.sym.transpose(data, axes=(0,2,3,1))
        sym_nhwc = mx.sym.Pooling(sym_nhwc, layout='NHWC', **kwargs)
        sym_nhwc = mx.sym.transpose(sym_nhwc, axes=(0,3,1,2), name='pool')
        return [sym, sym_nhwc]

    # While the float32 and float64 output is reliably consistent, float16 departs occasionally.
    # We compare nhwc and nchw results only within a given precision.
    for in_shape in [(3, 4, 8, 8), (2, 2, 20, 20)]:
        for kernel in [(2,2), (3,3), (4,4)]:
            for stride in [(1,1), (1,2), (2,1), (2,2)]:
                for data_type in [np.float64, np.float32, np.float16]:
                    ctx_list = [{'ctx': mx.gpu(0), 'pool_data': in_shape,
                                 'type_dict': {'pool_data': data_type}}]
                    symlist = make_pooling_syms(kernel=kernel, pool_type='max', stride=stride,
                                                pooling_convention='valid', name='pool')
                    check_consistency_NxM(symlist, ctx_list)

                    symlist = make_pooling_syms(kernel=kernel, pool_type='max', stride=stride,
                                                pooling_convention='full', name='pool')
                    check_consistency_NxM(symlist, ctx_list)

                    symlist = make_pooling_syms(kernel=(300,300), pool_type='max',
                                                global_pool=True, name='pool')
                    check_consistency_NxM(symlist, ctx_list)


@pytest.mark.serial
def test_pooling_with_type():
    ctx_list = [{'ctx': mx.gpu(0), 'pool_data': (2, 2, 10, 10), 'type_dict': {'pool_data': np.float64}},
                {'ctx': mx.gpu(0), 'pool_data': (2, 2, 10, 10), 'type_dict': {'pool_data': np.float32}},
                {'ctx': mx.gpu(0), 'pool_data': (2, 2, 10, 10), 'type_dict': {'pool_data': np.float16}},
                {'ctx': mx.cpu(0), 'pool_data': (2, 2, 10, 10), 'type_dict': {'pool_data': np.float64}},
                {'ctx': mx.cpu(0), 'pool_data': (2, 2, 10, 10), 'type_dict': {'pool_data': np.float32}}]
    sym = mx.sym.Pooling(kernel=(3,3), pool_type='max', pooling_convention='valid', name='pool')
    check_consistency(sym, ctx_list, rand_type=np.float16)

    sym = mx.sym.Pooling(kernel=(3,3), pool_type='max', pooling_convention='full', name='pool')
    check_consistency(sym, ctx_list, rand_type=np.float16)

    sym = mx.sym.Pooling(kernel=(300,300), pool_type='max', global_pool=True, name='pool')
    check_consistency(sym, ctx_list, rand_type=np.float16)


@pytest.mark.serial
def test_deconvolution_with_type():
    # Test basic deconvolution without exercising stride, pad or dilation.
    # 1D deconvolution
    sym = mx.sym.Deconvolution(num_filter=3, kernel=(3,), name='deconv')
    ctx_list = [{'ctx': mx.gpu(0), 'deconv_data': (2, 2, 7), 'type_dict': {'deconv_data': np.float64}},
                {'ctx': mx.gpu(0), 'deconv_data': (2, 2, 7), 'type_dict': {'deconv_data': np.float32}},
                {'ctx': mx.gpu(0), 'deconv_data': (2, 2, 7), 'type_dict': {'deconv_data': np.float16}},
                {'ctx': mx.cpu(0), 'deconv_data': (2, 2, 7), 'type_dict': {'deconv_data': np.float64}},
                {'ctx': mx.cpu(0), 'deconv_data': (2, 2, 7), 'type_dict': {'deconv_data': np.float32}}]
    # wider tolerance needed for true-fp16 test above
    tol = {np.dtype(np.float16): 0.3,
               np.dtype(np.float32): 1e-3,
               np.dtype(np.float64): 1e-5,
               np.dtype(np.uint8): 0,
               np.dtype(np.int32): 0}
    check_consistency(sym, ctx_list, rtol=tol, atol=tol)
    check_consistency(sym, ctx_list, rtol=tol, atol=tol, grad_req="add")

    # 2D deconvolution
    sym = mx.sym.Deconvolution(num_filter=2, kernel=(3,3), name='deconv')
    ctx_list = [{'ctx': mx.gpu(0), 'deconv_data': (2, 2, 10, 10), 'type_dict': {'deconv_data': np.float64}},
                {'ctx': mx.gpu(0), 'deconv_data': (2, 2, 10, 10), 'type_dict': {'deconv_data': np.float32}},
                {'ctx': mx.gpu(0), 'deconv_data': (2, 2, 10, 10), 'type_dict': {'deconv_data': np.float16}},
                {'ctx': mx.cpu(0), 'deconv_data': (2, 2, 10, 10), 'type_dict': {'deconv_data': np.float64}},
                {'ctx': mx.cpu(0), 'deconv_data': (2, 2, 10, 10), 'type_dict': {'deconv_data': np.float32}}]
    # wider tolerance needed for true-fp16 test above
    tol = {np.dtype(np.float16): 0.3,
               np.dtype(np.float32): 1e-3,
               np.dtype(np.float64): 1e-5,
               np.dtype(np.uint8): 0,
               np.dtype(np.int32): 0}
    check_consistency(sym, ctx_list, rtol=tol, atol=tol)
    check_consistency(sym, ctx_list, rtol=tol, atol=tol, grad_req="add")


@pytest.mark.serial
def test_deconvolution_options():

    # 1D deconvolution
    ctx_list = [{'ctx': mx.gpu(0), 'deconv_data': (2, 2, 7), 'type_dict': {'deconv_data': np.float64}},
                {'ctx': mx.gpu(0), 'deconv_data': (2, 2, 7), 'type_dict': {'deconv_data': np.float32}},
                {'ctx': mx.gpu(0), 'deconv_data': (2, 2, 7), 'type_dict': {'deconv_data': np.float16}},
                {'ctx': mx.cpu(0), 'deconv_data': (2, 2, 7), 'type_dict': {'deconv_data': np.float64}},
                {'ctx': mx.cpu(0), 'deconv_data': (2, 2, 7), 'type_dict': {'deconv_data': np.float32}}]
    # Pad > 0
    sym = mx.sym.Deconvolution(layout='NCW', num_filter=3, kernel=(3,), pad=(1,), name='deconv')
    sym_no_cudnn = mx.sym.Deconvolution(num_filter=3, kernel=(3,), pad=(1,), cudnn_off=True, name='deconv')
    check_consistency_NxM([sym, sym_no_cudnn], ctx_list)
    # Stride > 1
    sym = mx.sym.Deconvolution(layout='NCW', num_filter=3, kernel=(3,), stride=(2,), name='deconv')
    sym_no_cudnn = mx.sym.Deconvolution(num_filter=3, kernel=(3,), stride=(2,), cudnn_off=True, name='deconv')
    check_consistency_NxM([sym, sym_no_cudnn], ctx_list)
    # Dilate > 1
    sym = mx.sym.Deconvolution(layout='NCW', num_filter=3, kernel=(3,), dilate=(2,), name='deconv')
    sym_no_cudnn = mx.sym.Deconvolution(num_filter=3, kernel=(3,), dilate=(2,), cudnn_off=True, name='deconv')
    check_consistency_NxM([sym, sym_no_cudnn], ctx_list)

    # 2D deconvolution
    ctx_list = [{'ctx': mx.gpu(0), 'deconv_data': (2, 8, 10, 10), 'type_dict': {'deconv_data': np.float64}},
                {'ctx': mx.gpu(0), 'deconv_data': (2, 8, 10, 10), 'type_dict': {'deconv_data': np.float32}},
                {'ctx': mx.gpu(0), 'deconv_data': (2, 8, 10, 10), 'type_dict': {'deconv_data': np.float16}},
                {'ctx': mx.cpu(0), 'deconv_data': (2, 8, 10, 10), 'type_dict': {'deconv_data': np.float64}},
                {'ctx': mx.cpu(0), 'deconv_data': (2, 8, 10, 10), 'type_dict': {'deconv_data': np.float32}}]
    # Pad > 0
    sym = mx.sym.Deconvolution(num_filter=2, kernel=(3,3), pad=(1,1), name='deconv')
    sym_no_cudnn = mx.sym.Deconvolution(num_filter=2, kernel=(3,3), pad=(1,1), cudnn_off=True, name='deconv')
    check_consistency_NxM([sym, sym_no_cudnn], ctx_list)
    # Stride > 1
    sym = mx.sym.Deconvolution(num_filter=2, kernel=(3,3), stride=(2,2), name='deconv')
    sym_no_cudnn = mx.sym.Deconvolution(num_filter=2, kernel=(3,3), stride=(2,2), cudnn_off=True, name='deconv')
    check_consistency_NxM([sym, sym_no_cudnn], ctx_list)
    # Dilate > 1
    sym = mx.sym.Deconvolution(num_filter=2, kernel=(3,3), dilate=(2,2), name='deconv')
    sym_no_cudnn = mx.sym.Deconvolution(num_filter=2, kernel=(3,3), dilate=(2,2), cudnn_off=True, name='deconv')
    check_consistency_NxM([sym, sym_no_cudnn], ctx_list)

#    # 3D deconvolution (not yet enabled)
#    ctx_list = [{'ctx': mx.cpu(0), 'conv_data': (2, 2, 5, 7, 7), 'type_dict': {'conv_data': np.float64}},
#                {'ctx': mx.cpu(0), 'conv_data': (2, 2, 5, 7, 7), 'type_dict': {'conv_data': np.float64}},
#                {'ctx': mx.gpu(0), 'conv_data': (2, 2, 5, 7, 7), 'type_dict': {'conv_data': np.float64}},
#                {'ctx': mx.gpu(0), 'conv_data': (2, 2, 5, 7, 7), 'type_dict': {'conv_data': np.float32}}]
#    # Pad > 0
#    sym = mx.sym.Convolution(num_filter=3, kernel=(2,3,3), pad=(1,1,1), name='conv')
#    sym_no_cudnn = mx.sym.Convolution(num_filter=3, kernel=(2,3,3), pad=(1,1,1), cudnn_off=True, name='conv')
#    check_consistency_NxM([sym, sym_no_cudnn], ctx_list)
#    # Stride > 1
#    sym = mx.sym.Convolution(num_filter=3, kernel=(2,3,3), stride=(2,2,2), name='conv')
#    sym_no_cudnn = mx.sym.Convolution(num_filter=3, kernel=(2,3,3), stride=(2,2,2), cudnn_off=True, name='conv')
#    check_consistency_NxM([sym, sym_no_cudnn], ctx_list)


@pytest.mark.seed(1234)
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
    scale = 1
    ctx_list = [{'ctx': mx.gpu(0), 'data': (3, 6), 'type_dict': {'data': np.float32}},
                {'ctx': mx.cpu(0), 'data': (3, 6), 'type_dict': {'data': np.float32}}]
    check_consistency(sym, ctx_list, scale=scale)
    check_consistency(sym, ctx_list, scale=scale, grad_req="add")
    sym = mx.sym.GridGenerator(data=data, transform_type='warp', target_shape=(20, 20))
    ctx_list = [{'ctx': mx.gpu(0), 'data': (3, 2, 20, 20), 'type_dict': {'data': np.float32}},
                {'ctx': mx.cpu(0), 'data': (3, 2, 20, 20), 'type_dict': {'data': np.float32}}]
    check_consistency(sym, ctx_list)
    check_consistency(sym, ctx_list, grad_req="add")


def test_spatial_transformer_with_type():
    data = mx.sym.Variable('data')
    loc = mx.sym.Flatten(data)
    loc = mx.sym.FullyConnected(data=loc, num_hidden=10)
    loc = mx.sym.Activation(data=loc, act_type='relu')
    loc = mx.sym.FullyConnected(data=loc, num_hidden=6)
    sym = mx.sym.SpatialTransformer(data=data, loc=loc, target_shape=(10, 10),
                                    transform_type="affine", sampler_type="bilinear", cudnn_off=True)
    ctx_list = [{'ctx': mx.gpu(0), 'data': (1, 5, 10, 10), 'type_dict': {'data': np.float64}},
                {'ctx': mx.cpu(0), 'data': (1, 5, 10, 10), 'type_dict': {'data': np.float64}}]
    check_consistency(sym, ctx_list)
    check_consistency(sym, ctx_list, grad_req="add")
    sym = mx.sym.SpatialTransformer(data=data, loc=loc, target_shape=(10, 10),
                                    transform_type="affine", sampler_type="bilinear", cudnn_off=False)
    check_consistency(sym, ctx_list)
    check_consistency(sym, ctx_list, grad_req="add")

def test_pooling_with_type2():
    # While the float32 and float64 output is reliably consistent, float16 departs occasionally.
    # We compare cpu and gpu results only within a given precision.
    for data_type in [np.float64, np.float32, np.float16]:
        ctx_list = [{'ctx': mx.gpu(0), 'pool_data': (10, 2, 10, 10), 'type_dict': {'pool_data': data_type}},
                    {'ctx': mx.cpu(0), 'pool_data': (10, 2, 10, 10), 'type_dict': {'pool_data': data_type}}]

        sym = mx.sym.Pooling(name='pool', kernel=(3,3), stride=(2,2), pool_type='max')
        check_consistency(sym, ctx_list)

        sym = mx.sym.Pooling(name='pool', kernel=(3,3), pad=(1,1), pool_type='avg')
        check_consistency(sym, ctx_list)

        sym = mx.sym.Pooling(name='pool', kernel=(5,5), pad=(2,2), pool_type='max')
        check_consistency(sym, ctx_list)

        sym = mx.sym.Pooling(name='pool', kernel=(3,3), pad=(1,1), pool_type='sum')
        check_consistency(sym, ctx_list)

def test_pooling_nhwc_with_type():
    def make_pooling_syms(**kwargs):
        # Conventional NCHW layout pooling
        sym = mx.sym.Pooling(**kwargs)
        # NHWC pooling
        data = mx.sym.Variable('pool_data')
        sym_nhwc = mx.sym.transpose(data, axes=(0,2,3,1))
        sym_nhwc = mx.sym.Pooling(sym_nhwc, layout='NHWC', **kwargs)
        sym_nhwc = mx.sym.transpose(sym_nhwc, axes=(0,3,1,2), name='pool')
        return [sym, sym_nhwc]

    # While the float32 and float64 output is reliably consistent, float16 departs occasionally.
    # We compare nhwc and nchw results only within a given precision.
    for data_type in [np.float64, np.float32, np.float16]:
        # NHWC pooling only enabled on GPU with CUDNN
        ctx_list = [{'ctx': mx.gpu(0), 'pool_data': (10, 2, 10, 10), 'type_dict': {'pool_data': data_type}}]
        symlist = make_pooling_syms(name='pool', kernel=(3,3), stride=(2,2), pool_type='max')
        check_consistency_NxM(symlist, ctx_list)

        symlist = make_pooling_syms(name='pool', kernel=(3,3), pad=(1,1), pool_type='avg')
        check_consistency_NxM(symlist, ctx_list)

        symlist = make_pooling_syms(name='pool', kernel=(5,5), pad=(2,2), pool_type='max')
        check_consistency_NxM(symlist, ctx_list)


@pytest.mark.serial
def test_pooling_versions():

    # Produce the name of the 'transposed' layout, given the dimension
    def transposed_layout(ndim):
        if ndim < 3 or ndim > 5:
            raise RuntimeError("Invalid data dim, expecting 3, 4 or 5")
        return ('NWC', 'NHWC', 'NDHWC')[ndim-3]

    # default padding is all zeros
    def is_default_pad(pad):
        return pad == (0,) * len(pad)

    # default stride is all ones
    def is_default_stride(stride):
        return stride == (1,) * len(stride)

    # returns True/False randomly with equal probability
    def random_choice():
        return np.random.random(1)[0] < 0.5

    def test_pooling_versions_helper(pool_op_list, data, kernel, pool_type, pad, stride,
                                     pooling_convention='valid', global_pool=False, p_value=2,
                                     count_include_pad=True, tol=None, dtype=np.float32):
        ctx_list = []
        sym_list = []
        for pool_ctx in pool_op_list:
            (pool_op, ctx_type) = pool_ctx.rsplit('_', 1)
            expected_ctxs = ['cpu', 'gpu', 'cudnn']
            if ctx_type not in expected_ctxs:
                raise RuntimeError('Expected one of {}, saw {}.'.format(expected_ctxs, ctx_type))
            ctx = mx.cpu(0) if ctx_type == 'cpu' else mx.gpu(0)
            ctx_list.append({'ctx': ctx, 'pool_data': data, 'type_dict': {'pool_data': dtype}})
            # start with pool args present in all cases
            pool_op_args = {'kernel': kernel, 'pool_type': pool_type,
                            'pooling_convention' : pooling_convention, 'name' : 'pool'}
            # add other args as needed
            if global_pool:
                pool_op_args['global_pool'] = True
            else:
                # Add pad and stride param if needed, plus randomly when it matches the default
                if not is_default_pad(pad) or random_choice():
                    pool_op_args.update({'pad' : pad})
                if not is_default_stride(stride) or random_choice():
                    pool_op_args.update({'stride' : stride})

            expected_pool_ops = ['pool', 'pool_transposed']
            pool_op_args.update({'p_value' : p_value, 'count_include_pad' : count_include_pad})
            if ctx_type != 'cpu':
                pool_op_args['cudnn_off'] = ctx_type == 'gpu'
            if pool_op == 'pool':
                # isolate pooling input from symbol input to test shared tensor optimizations
                buffered_input = mx.sym.identity(name='pool')
                sym = mx.sym.Pooling(buffered_input, **pool_op_args)
            elif pool_op == 'pool_transposed':
                ndim = len(data)
                # NCW->NWC axes=(0,2,1) NCHW->NHWC axes=(0,2,3,1) NCDHW->NDHWC axes=(0,2,3,4,1);
                axes = (0,) + tuple(range(2,ndim)) + (1,)
                transposed = mx.sym.transpose(axes=axes, name='pool')
                pooled = mx.sym.Pooling(data=transposed, layout=transposed_layout(ndim),
                                        **pool_op_args)
                # NWC->NCW axes=(0,2,1) NHWC->NCHW axes=(0,3,1,2) NDHWC->NCDHW axes=(0,4,1,2,3);
                axes = (0, ndim-1) + tuple(range(1,ndim-1))
                sym = mx.sym.transpose(data=pooled, axes=axes, name='pool')
            else:
                raise RuntimeError('Expected one of {}, saw {}.'.format(expected_pool_ops,
                                                                        pool_op))
            sym_list.append(sym)

        check_consistency(sym_list, ctx_list, equal_nan=(not count_include_pad), rtol=tol, atol=tol)

    def test_pooling_dim(dim, pool_type, dtype, pool_op_list, p_value=2, count_include_pad=True,
                         tol=None):
        if dim == '1D':
            data = (3, 3, 10)
            kernels = [(4,), (4,), (5,)]
            pads = [(0,), (2,), (2,)]
            strides = [(1,), (2,), (1,)]
        elif dim == '2D_no_padding':
            data = (3, 2, 20, 20)
            kernels = [(3, 3), (4, 5)]
            pads = [(0, 0), (0, 0)]
            strides = [(1, 1), (2, 1)]
        elif dim == '2D':
            data = (2, 2, 20, 20)
            kernels = [(3, 3), (3, 5), (4, 5), (4, 5)]
            pads = [(0, 0), (1, 2), (0, 0), (2, 3)]
            strides = [(1, 1), (1, 1), (2, 1), (1, 1)]
        elif dim == '3D':
            data = (2, 3, 20, 20, 20)
            kernels = [(4, 5, 3), (4, 5, 3), (3, 5, 7)]
            pads = [(0, 0, 0), (2, 3, 2), (1, 2, 3)]
            strides = [(1, 1, 1), (2, 3, 1), (1, 1, 1)]
        else:
            raise RuntimeError('Unexpected pooling test class: {}.'.format(dim))

        for kernel, pad, stride in zip(kernels, pads, strides):
            for pooling_convention in ['valid', 'full']:
                try:
                    test_pooling_versions_helper(pool_op_list=pool_op_list,
                                     data=data, kernel=kernel, pad=pad, stride=stride,
                                     pool_type=pool_type, pooling_convention=pooling_convention,
                                     global_pool=False, p_value=p_value,
                                     count_include_pad=count_include_pad, tol=tol, dtype=dtype)
                except:
                    print('pool_op_list = {}'.format(pool_op_list))
                    print('kernel={}, pad={}, stride={}'.format(kernel, pad, stride))
                    print('pool_type={}, pooling_convention={}, global_pool=False'.format(pool_type,
                          pooling_convention))
                    print('p_value={}, count_include_pad={}, dtype={}'.format(p_value,
                          count_include_pad, dtype))
                    print('environ = \n{}'.format(os.environ))
                    raise

        # Make sure kernel is ignored during global_pool by sometimes setting it to a crazy value
        kernel = kernels[0]
        if random_choice():
            kernel = (300,) * len(kernel)

        test_pooling_versions_helper(pool_op_list=pool_op_list,
                                     data=data, kernel=kernel, pad=None, stride=None,
                                     pool_type=pool_type, global_pool=True, p_value=p_value,
                                     count_include_pad=count_include_pad, tol=tol, dtype=dtype)

    # The various implementations of the standard pooling operator
    std_pool_op_list = ['pool_cpu', 'pool_transposed_cpu',
                        'pool_gpu', 'pool_transposed_gpu',
                        'pool_cudnn', 'pool_transposed_cudnn']

    for dtype in [np.float32, np.float64, np.float16]:
        # Testing of the standard (not 'v1') pooling operator is universal across all
        # data dimensions, implementations and layouts.
        for dim in ['1D', '2D', '3D']:
            test_pooling_dim(dim, 'max', dtype, std_pool_op_list)
            test_pooling_dim(dim, 'avg', dtype, std_pool_op_list, count_include_pad=True)
            test_pooling_dim(dim, 'avg', dtype, std_pool_op_list, count_include_pad=False)
            test_pooling_dim(dim, 'sum', dtype, std_pool_op_list)
            test_pooling_dim(dim, 'lp', dtype, std_pool_op_list, p_value=1)
            test_pooling_dim(dim, 'lp', dtype, std_pool_op_list, p_value=2)
            test_pooling_dim(dim, 'lp', dtype, std_pool_op_list, p_value=3)


def test_pooling_full_2d():
    def test_pooling_full_2d_type(pool_type):
        data = (2, 2, 10, 10)
        kernel = (4, 5)
        pad = (1, 2)
        stride = (3, 4)

        convention = 'full'
        ctx_list = []
        sym_list = []

        # o_h = ceil((10 + 1 + 1 - 4) / 3) + 1 = 4
        # o_w = ceil((10 + 2 + 2 - 5) / 4) + 1 = 4
        ctx_list.append({'ctx': mx.cpu(0), 'pool_data': data, 'type_dict': {'pool_data': np.float32}})
        sym_list.append(mx.sym.Pooling(kernel=kernel, pad=pad, stride=stride, pool_type=pool_type,
                                       pooling_convention=convention, global_pool=False, name='pool'))

        ctx_list.append({'ctx': mx.gpu(0), 'pool_data': data, 'type_dict': {'pool_data': np.float32}})
        sym_list.append(mx.sym.Pooling(kernel=kernel, pad=pad, stride=stride, pool_type=pool_type,
                                       pooling_convention=convention, global_pool=False, name='pool'))

        check_consistency(sym_list, ctx_list)

    test_pooling_full_2d_type('max')
    test_pooling_full_2d_type('avg')
    test_pooling_full_2d_type('sum')


@pytest.mark.serial
def test_flatten_slice_after_conv():
    ctx_list = []

    data = mx.sym.Variable('conv_data')
    conv = mx.symbol.Convolution(data=data, name='conv', num_filter=16, kernel=(3,3), stride=(1,1))
    flatten = mx.symbol.flatten(data=conv)
    slice_sym = mx.symbol.slice(data=flatten, begin=0, end=1)

    ctx_list = [{'ctx': mx.gpu(0), 'conv_data': (2, 16, 16, 16), 'type_dict': {'conv_data': np.float32}},
                {'ctx': mx.cpu(0), 'conv_data': (2, 16, 16, 16), 'type_dict': {'conv_data': np.float32}}]
    check_consistency(slice_sym, ctx_list, scale=0.5)


def test_bilinear_resize_op():
    ctx_list = [{'ctx': mx.cpu(0), 'data': (2, 2, 20, 20), 'type_dict': {'data': np.float32}},
                {'ctx': mx.gpu(0), 'data': (2, 2, 20, 20), 'type_dict': {'data': np.float32}}]

    data = mx.sym.Variable('data')
    sym = mx.sym.contrib.BilinearResize2D(data, height=10, width=5, align_corners=True)
    check_consistency(sym, ctx_list)

    sym = mx.sym.contrib.BilinearResize2D(data, height=10, width=5, align_corners=False)
    check_consistency(sym, ctx_list)

    sym = mx.sym.contrib.BilinearResize2D(data, None, scale_height=2, scale_width=0.5, mode='odd_scale', align_corners=True)
    check_consistency(sym, ctx_list)

    sym = mx.sym.contrib.BilinearResize2D(data, None, scale_height=2, scale_width=0.5, mode='odd_scale', align_corners=False)
    check_consistency(sym, ctx_list)

    sym = mx.sym.contrib.BilinearResize2D(data, None, scale_height=0.5, scale_width=2, mode='to_even_up', align_corners=True)
    check_consistency(sym, ctx_list)

    sym = mx.sym.contrib.BilinearResize2D(data, None, scale_height=0.5, scale_width=2, mode='to_even_up', align_corners=False)
    check_consistency(sym, ctx_list)

@pytest.mark.serial
def test_global_pooling():
    def test_1d_pooling(pool_type, p_value=2):
        data = (2, 3, 20)
        kernel = (4,)
        pad = (2,)
        stride = (2,)

        ctx_list = []
        sym_list = []

        pooling_convention = 'valid'

        ctx_list.append({'ctx': mx.cpu(0), 'pool_data': data, 'type_dict': {'pool_data': np.float32}})
        sym_list.append(mx.sym.Pooling(kernel=kernel, pad=pad, stride=stride, pool_type=pool_type,
                                       pooling_convention=pooling_convention, global_pool=True, name='pool', p_value=p_value))

        ctx_list.append({'ctx': mx.cpu(0), 'pool_data': data, 'type_dict': {'pool_data': np.float32}})
        sym_list.append(mx.sym.Pooling(kernel=kernel, pool_type=pool_type,
                                       pooling_convention=pooling_convention, global_pool=True, name='pool', p_value=p_value))

        ctx_list.append({'ctx': mx.cpu(0), 'pool_data': data, 'type_dict': {'pool_data': np.float32}})
        sym_list.append(mx.sym.Pooling(pool_type=pool_type,
                                       pooling_convention=pooling_convention, global_pool=True, name='pool', p_value=p_value))

        ctx_list.append({'ctx': mx.gpu(0), 'pool_data': data, 'type_dict': {'pool_data': np.float32}})
        sym_list.append(mx.sym.Pooling(kernel=kernel, pad=pad, stride=stride, pool_type=pool_type,
                                       pooling_convention=pooling_convention, global_pool=True, p_value=p_value, cudnn_off=False, name='pool'))

        ctx_list.append({'ctx': mx.gpu(0), 'pool_data': data, 'type_dict': {'pool_data': np.float32}})
        sym_list.append(mx.sym.Pooling(kernel=kernel, pool_type=pool_type,
                                       pooling_convention=pooling_convention, global_pool=True, p_value=p_value, cudnn_off=False, name='pool'))

        ctx_list.append({'ctx': mx.gpu(0), 'pool_data': data, 'type_dict': {'pool_data': np.float32}})
        sym_list.append(mx.sym.Pooling(pool_type=pool_type,
                                       pooling_convention=pooling_convention, global_pool=True, p_value=p_value, cudnn_off=False, name='pool'))

        ctx_list.append({'ctx': mx.gpu(0), 'pool_data': data, 'type_dict': {'pool_data': np.float32}})
        sym_list.append(mx.sym.Pooling(kernel=kernel, pad=pad, stride=stride, pool_type=pool_type,
                                       pooling_convention=pooling_convention, global_pool=True, p_value=p_value, cudnn_off=True, name='pool'))

        ctx_list.append({'ctx': mx.gpu(0), 'pool_data': data, 'type_dict': {'pool_data': np.float32}})
        sym_list.append(mx.sym.Pooling(kernel=kernel, pool_type=pool_type,
                                       pooling_convention=pooling_convention, global_pool=True, p_value=p_value, cudnn_off=True, name='pool'))

        ctx_list.append({'ctx': mx.gpu(0), 'pool_data': data, 'type_dict': {'pool_data': np.float32}})
        sym_list.append(mx.sym.Pooling(pool_type=pool_type,
                                       pooling_convention=pooling_convention, global_pool=True, p_value=p_value, cudnn_off=True, name='pool'))

        check_consistency(sym_list, ctx_list)

    def test_2d_pooling(pool_type, p_value=2):
        data = (2, 3, 20, 20)
        kernel = (4, 4)
        pad = (2, 2)
        stride = (2, 2)

        ctx_list = []
        sym_list = []

        pooling_convention = 'valid'

        ctx_list.append({'ctx': mx.cpu(0), 'pool_data': data, 'type_dict': {'pool_data': np.float32}})
        sym_list.append(mx.sym.Pooling(kernel=kernel, pad=pad, stride=stride, pool_type=pool_type,
                                       pooling_convention=pooling_convention, global_pool=True, p_value=p_value, name='pool'))

        ctx_list.append({'ctx': mx.cpu(0), 'pool_data': data, 'type_dict': {'pool_data': np.float32}})
        sym_list.append(mx.sym.Pooling(kernel=kernel, pool_type=pool_type,
                                       pooling_convention=pooling_convention, global_pool=True, p_value=p_value, name='pool'))

        ctx_list.append({'ctx': mx.cpu(0), 'pool_data': data, 'type_dict': {'pool_data': np.float32}})
        sym_list.append(mx.sym.Pooling(pool_type=pool_type,
                                       pooling_convention=pooling_convention, global_pool=True, p_value=p_value, name='pool'))

        ctx_list.append({'ctx': mx.gpu(0), 'pool_data': data, 'type_dict': {'pool_data': np.float32}})
        sym_list.append(mx.sym.Pooling(kernel=kernel, pad=pad, stride=stride, pool_type=pool_type,
                                       pooling_convention=pooling_convention, global_pool=True, p_value=p_value, cudnn_off=False, name='pool'))

        ctx_list.append({'ctx': mx.gpu(0), 'pool_data': data, 'type_dict': {'pool_data': np.float32}})
        sym_list.append(mx.sym.Pooling(kernel=kernel, pool_type=pool_type,
                                       pooling_convention=pooling_convention, global_pool=True, p_value=p_value, cudnn_off=False, name='pool'))

        ctx_list.append({'ctx': mx.gpu(0), 'pool_data': data, 'type_dict': {'pool_data': np.float32}})
        sym_list.append(mx.sym.Pooling(pool_type=pool_type,
                                       pooling_convention=pooling_convention, global_pool=True, p_value=p_value, cudnn_off=False, name='pool'))

        ctx_list.append({'ctx': mx.gpu(0), 'pool_data': data, 'type_dict': {'pool_data': np.float32}})
        sym_list.append(mx.sym.Pooling(kernel=kernel, pad=pad, stride=stride, pool_type=pool_type,
                                       pooling_convention=pooling_convention, global_pool=True, p_value=p_value, cudnn_off=True, name='pool'))

        ctx_list.append({'ctx': mx.gpu(0), 'pool_data': data, 'type_dict': {'pool_data': np.float32}})
        sym_list.append(mx.sym.Pooling(kernel=kernel, pool_type=pool_type,
                                       pooling_convention=pooling_convention, global_pool=True, p_value=p_value, cudnn_off=True, name='pool'))

        ctx_list.append({'ctx': mx.gpu(0), 'pool_data': data, 'type_dict': {'pool_data': np.float32}})
        sym_list.append(mx.sym.Pooling(pool_type=pool_type,
                                       pooling_convention=pooling_convention, global_pool=True, p_value=p_value, cudnn_off=True, name='pool'))


        check_consistency(sym_list, ctx_list)

    test_1d_pooling('max')
    test_1d_pooling('avg')
    test_1d_pooling('sum')
    test_1d_pooling('lp', p_value=1)
    test_1d_pooling('lp', p_value=2)
    test_1d_pooling('lp', p_value=3)

    test_2d_pooling('max')
    test_2d_pooling('avg')
    test_2d_pooling('sum')
    test_2d_pooling('lp', p_value=1)
    test_2d_pooling('lp', p_value=2)
    test_2d_pooling('lp', p_value=3)


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
    dev_types = [[mx.gpu(0), [np.float64, np.float32, np.float16]],
                 [mx.cpu(0), [np.float64, np.float32]] ]
    for num_args in range(1, 6):
        ews_arg_shape = {}
        for i in range(num_args):
            ews_arg_shape['ews_arg'+str(i)] = (2, 10)
        sym = mx.sym.ElementWiseSum(name='ews', num_args=num_args)
        ctx_list = []
        for dev, types in dev_types:
            for dtype in types:
                ews_arg_dtype = {'type_dict':{}}
                for i in range(num_args):
                    ews_arg_dtype['type_dict']['ews_arg'+str(i)] = dtype
                ctx_elem = {'ctx': dev}
                ctx_elem.update(ews_arg_shape)
                ctx_elem.update(ews_arg_dtype)
                ctx_list.append(ctx_elem)
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
    # Sizes are divisible by 8 to test TensorCore on Volta GPU.
    sym = mx.sym.FullyConnected(num_hidden=8, name='inner')
    ctx_list = [{'ctx': mx.gpu(0), 'inner_data': (16, 24), 'type_dict': {'inner_data': np.float16}},
                {'ctx': mx.cpu(0), 'inner_data': (16, 24), 'type_dict': {'inner_data': np.float32}}]
    check_consistency(sym, ctx_list)


def test_activation_with_type():
    act_types = ['relu', 'sigmoid', 'tanh', 'softrelu', 'softsign']
    shape = (2, 2, 10, 10)
    for act_type in act_types:
        sym = mx.sym.Activation(name='act', act_type=act_type)
        ctx_list = [{'ctx': mx.gpu(0), 'act_data': shape, 'type_dict': {'act_data': np.float64}},
                    {'ctx': mx.gpu(0), 'act_data': shape, 'type_dict': {'act_data': np.float32}},
                    {'ctx': mx.gpu(0), 'act_data': shape, 'type_dict': {'act_data': np.float16}},
                    {'ctx': mx.cpu(0), 'act_data': shape, 'type_dict': {'act_data': np.float64}},
                    {'ctx': mx.cpu(0), 'act_data': shape, 'type_dict': {'act_data': np.float32}},
                    {'ctx': mx.cpu(0), 'act_data': shape, 'type_dict': {'act_data': np.float16}}]
        check_consistency(sym, ctx_list)


def test_lrn():
    sym = mx.sym.LRN(alpha=0.0001, beta=0.75, knorm=2, nsize=5, name='lrn')
    ctx_list = [{'ctx': mx.gpu(0), 'lrn_data': (2, 6, 10, 10), 'type_dict': {'lrn_data': np.float32}},
                {'ctx': mx.cpu(0), 'lrn_data': (2, 6, 10, 10), 'type_dict': {'lrn_data': np.float32}}]
    check_consistency(sym, ctx_list)


@pytest.mark.skipif(os.environ.get('MXNET_ENGINE_TYPE') == 'NaiveEngine',
                    reason="Testing with naive engine consistently triggers illegal memory access. Tracked in #17713")
def test_embedding_with_type():
    def test_embedding_helper(data_types, weight_types, low_pad, high_pad):
        NVD = [[20, 10, 20], [200, 10, 300], [10000, 4, 20]]
        for safe_accumulation in ['0', '1', None]:
            for N, V, D in NVD:
                with environment('MXNET_SAFE_ACCUMULATION', safe_accumulation):
                    if N > 1000 and safe_accumulation != '1':
                        break
                    sym = mx.sym.Embedding(name='embedding', input_dim=V, output_dim=D)
                    ctx_list = []
                    for data_type in data_types:
                        for weight_type in weight_types:
                            ctx_list.append({'ctx': mx.gpu(0), 'embedding_data': (N,),
                                'type_dict': {'embedding_data': data_type, 'embedding_weight': weight_type}})
                            ctx_list.append({'ctx': mx.cpu(0), 'embedding_data': (N,),
                                'type_dict': {'embedding_data': data_type, 'embedding_weight': weight_type}})
                    arg_params = {'embedding_data': np.random.randint(low=-low_pad, high=V+high_pad, size=(N,))}
                    check_consistency(sym, ctx_list, grad_req={'embedding_data': 'null','embedding_weight': 'write'},
                                      arg_params=arg_params, scale=0.1)

    data_types = [np.float16, np.float32, np.float64, np.int32]
    weight_types = [np.float16, np.float32, np.float64]
    test_embedding_helper(data_types, weight_types, 5, 5)
    data_types = [np.uint8]
    weight_types = [np.float16, np.float32, np.float64]
    test_embedding_helper(data_types, weight_types, 0, 5)


def test_take_with_type():
    sym = mx.sym.take(name='take')
    for safe_accumulation in ['0', '1', None]:
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
                with environment('MXNET_SAFE_ACCUMULATION', safe_accumulation):
                    check_consistency(sym, ctx_list,
                                      grad_req={'take_indices': 'null',
                                                'take_a': 'write'},
                                      arg_params=arg_params)

    # check a large num of indices: may underflow calculating gradient in FP16,
    # if MXNET_SAFE_ACCUMULATION is not activated
    with environment('MXNET_SAFE_ACCUMULATION', '1'):
        data_size = 4
        indices_size = 10000
        out_dim = 20
        data_types = [np.float16, np.float32, np.float64]
        indices_types = [np.float16, np.float32, np.float64, np.int32]
        # axis 0
        sym = mx.sym.take(name='take', axis=0)
        ctx_list = []
        for data_type in data_types:
            for index_type in indices_types:
                ctx_list.append({'ctx': mx.cpu(0), 'take_indices': (indices_size,),
                    'take_a': (data_size, out_dim),
                    'type_dict': {'take_indices': index_type, 'take_a': data_type}})
                ctx_list.append({'ctx': mx.gpu(0), 'take_indices': (indices_size,),
                    'take_a': (data_size, out_dim),
                    'type_dict': {'take_indices': index_type, 'take_a': data_type}})
                arg_params = {'take_indices': np.random.randint(0, data_size,
                              size=(indices_size,)),
                              'take_a': np.random.normal(size=(data_size, out_dim))}
                check_consistency(sym, ctx_list,
                                  grad_req={'take_indices': 'null','take_a': 'write'},
                                  arg_params=arg_params)
        # axis 1
        sym = mx.sym.take(name='take', axis=1)
        ctx_list = []
        for data_type in data_types:
            for index_type in indices_types:
                ctx_list.append({'ctx': mx.cpu(0), 'take_indices': (indices_size,),
                    'take_a': (data_size, out_dim),
                    'type_dict': {'take_indices': index_type, 'take_a': data_type}})
                ctx_list.append({'ctx': mx.gpu(0), 'take_indices': (indices_size,),
                    'take_a': (data_size, out_dim),
                    'type_dict': {'take_indices': index_type, 'take_a': data_type}})
                arg_params = {'take_indices': np.random.randint(0, data_size,
                              size=(indices_size,)),
                              'take_a': np.random.normal(size=(data_size, out_dim))}
                check_consistency(sym, ctx_list,
                                  grad_req={'take_indices': 'null','take_a': 'write'},
                                  arg_params=arg_params)

@pytest.mark.serial
def test_psroipooling_with_type():
    arg_params = {
        'psroipool_rois': np.array([[0, 10, 22, 161, 173], [0, 20, 15, 154, 160]])}

    # plain psroipooling
    sym = mx.sym.contrib.PSROIPooling(spatial_scale=0.0625, output_dim=2, pooled_size=3, name='psroipool')
    ctx_list = [{'ctx': mx.gpu(0),
                 'psroipool_data': (1, 18, 14, 14),
                 'psroipool_rois': (2, 5),
                 'type_dict': {'psroipool_data': np.float64, 'psroipool_rois': np.float64}},
                {'ctx': mx.gpu(0),
                 'psroipool_data': (1, 18, 14, 14),
                 'psroipool_rois': (2, 5),
                 'type_dict': {'psroipool_data': np.float32, 'psroipool_rois': np.float32}},
                {'ctx': mx.gpu(0),
                 'psroipool_data': (1, 18, 14, 14),
                 'psroipool_rois': (2, 5),
                 'type_dict': {'psroipool_data': np.float16, 'psroipool_rois': np.float16}},
                ]

    check_consistency(sym, ctx_list, grad_req={'psroipool_data': 'write',
                                               'psroipool_rois': 'null'}, arg_params=arg_params)


@pytest.mark.serial
def test_deformable_psroipooling_with_type():
    tol = {np.dtype(np.float32): 1e-1,
           np.dtype(np.float64): 1e-3,
           np.dtype(np.float16): 1e-2}

    arg_params = {
        'deformable_psroipool_rois': np.array([[0, 10, 22, 161, 173], [0, 20, 15, 154, 160]])}

    # deformable psroipooling
    sym = mx.sym.contrib.DeformablePSROIPooling(spatial_scale=0.0625, sample_per_part=4, group_size=3, pooled_size=3,
                                                output_dim=2, trans_std=0.1, no_trans=False, name='deformable_psroipool')

    ctx_list = [{'ctx': mx.gpu(0),
                 'deformable_psroipool_data': (1, 18, 14, 14),
                 'deformable_psroipool_rois': (2, 5),
                 'deformable_psroipool_trans': (2, 4, 3, 3),
                 'type_dict': {'deformable_psroipool_data': np.float64, 'deformable_psroipool_rois': np.float64,
                               'deformable_psroipool_trans': np.float64}},
                {'ctx': mx.gpu(0),
                 'deformable_psroipool_data': (1, 18, 14, 14),
                 'deformable_psroipool_rois': (2, 5),
                 'deformable_psroipool_trans': (2, 4, 3, 3),
                 'type_dict': {'deformable_psroipool_data': np.float32, 'deformable_psroipool_rois': np.float32,
                               'deformable_psroipool_trans': np.float32}},
                {'ctx': mx.gpu(0),
                 'deformable_psroipool_data': (1, 18, 14, 14),
                 'deformable_psroipool_rois': (2, 5),
                 'deformable_psroipool_trans': (2, 4, 3, 3),
                 'type_dict': {'deformable_psroipool_data': np.float16, 'deformable_psroipool_rois': np.float16,
                               'deformable_psroipool_trans': np.float16}},
                {'ctx': mx.cpu(0),
                 'deformable_psroipool_data': (1, 18, 14, 14),
                 'deformable_psroipool_rois': (2, 5),
                 'deformable_psroipool_trans': (2, 4, 3, 3),
                 'type_dict': {'deformable_psroipool_data': np.float64, 'deformable_psroipool_rois': np.float64,
                               'deformable_psroipool_trans': np.float64}},
                {'ctx': mx.cpu(0),
                 'deformable_psroipool_data': (1, 18, 14, 14),
                 'deformable_psroipool_rois': (2, 5),
                 'deformable_psroipool_trans': (2, 4, 3, 3),
                 'type_dict': {'deformable_psroipool_data': np.float32, 'deformable_psroipool_rois': np.float32,
                               'deformable_psroipool_trans': np.float32}},
                {'ctx': mx.cpu(0),
                 'deformable_psroipool_data': (1, 18, 14, 14),
                 'deformable_psroipool_rois': (2, 5),
                 'deformable_psroipool_trans': (2, 4, 3, 3),
                 'type_dict': {'deformable_psroipool_data': np.float16, 'deformable_psroipool_rois': np.float16,
                               'deformable_psroipool_trans': np.float16}},
                ]

    check_consistency(sym, ctx_list, scale=0.1, rtol=tol, atol=tol,
                      grad_req={'deformable_psroipool_data': 'write',
                                'deformable_psroipool_rois': 'null',
                                'deformable_psroipool_trans': 'write'}, arg_params=arg_params)


@pytest.mark.serial
def test_deformable_convolution_with_type():
    tol = {np.dtype(np.float32): 1e-1,
           np.dtype(np.float64): 1e-3}

    sym = mx.sym.npx.deformable_convolution(num_filter=3, kernel=(3,3), name='deformable_conv')
    # since atomicAdd does not support fp16 (which deformable conv uses in backward), we do not test fp16 here
    ctx_list = [{'ctx': mx.gpu(0),
                 'deformable_conv_data': (2, 2, 10, 10),
                 'deformable_conv_offset': (2, 18, 8, 8),
                 'type_dict': {'deformable_conv_data': np.float64, 'deformable_conv_offset': np.float64}},
                {'ctx': mx.gpu(0),
                 'deformable_conv_data': (2, 2, 10, 10),
                 'deformable_conv_offset': (2, 18, 8, 8),
                 'type_dict': {'deformable_conv_data': np.float32, 'deformable_conv_offset': np.float32}},
                {'ctx': mx.cpu(0),
                 'deformable_conv_data': (2, 2, 10, 10),
                 'deformable_conv_offset': (2, 18, 8, 8),
                 'type_dict': {'deformable_conv_data': np.float64, 'deformable_conv_offset': np.float64}},
                {'ctx': mx.cpu(0),
                 'deformable_conv_data': (2, 2, 10, 10),
                 'deformable_conv_offset': (2, 18, 8, 8),
                 'type_dict': {'deformable_conv_data': np.float32, 'deformable_conv_offset': np.float32}},
                ]

    check_consistency(sym, ctx_list, scale=0.1, rtol=tol, atol=tol)
    # test ability to turn off training on bias
    check_consistency(sym, ctx_list, scale=0.1, rtol=tol, atol=tol,
                      grad_req={'deformable_conv_data': 'write',
                                'deformable_conv_offset': 'write',
                                'deformable_conv_weight': 'write',
                                'deformable_conv_bias': 'null'})


def test_deformable_convolution_options():
    tol = {np.dtype(np.float32): 1e-1,
           np.dtype(np.float64): 1e-3}
    # 2D convolution
    # since atomicAdd does not support fp16 (which deformable conv uses in backward), we do not test fp16 here

    # Pad > 0
    ctx_list = [{'ctx': mx.gpu(0),
                 'deformable_conv_data': (2, 2, 7, 7),
                 'deformable_conv_offset': (2, 18, 7, 7),
                 'type_dict': {'deformable_conv_data': np.float64, 'deformable_conv_offset': np.float64}},
                {'ctx': mx.gpu(0),
                 'deformable_conv_data': (2, 2, 7, 7),
                 'deformable_conv_offset': (2, 18, 7, 7),
                 'type_dict': {'deformable_conv_data': np.float32, 'deformable_conv_offset': np.float32}},
                {'ctx': mx.cpu(0),
                 'deformable_conv_data': (2, 2, 7, 7),
                 'deformable_conv_offset': (2, 18, 7, 7),
                 'type_dict': {'deformable_conv_data': np.float64, 'deformable_conv_offset': np.float64}},
                {'ctx': mx.cpu(0),
                 'deformable_conv_data': (2, 2, 7, 7),
                 'deformable_conv_offset': (2, 18, 7, 7),
                 'type_dict': {'deformable_conv_data': np.float32, 'deformable_conv_offset': np.float32}},
                ]
    sym = mx.sym.npx.deformable_convolution(num_filter=3, kernel=(3,3), pad=(1,1), name='deformable_conv')
    check_consistency(sym, ctx_list, scale=0.1, rtol=tol, atol=tol)

    # Stride > 1
    ctx_list = [{'ctx': mx.gpu(0),
                 'deformable_conv_data': (2, 2, 7, 7),
                 'deformable_conv_offset': (2, 18, 3, 3),
                 'type_dict': {'deformable_conv_data': np.float64, 'deformable_conv_offset': np.float64}},
                {'ctx': mx.gpu(0),
                 'deformable_conv_data': (2, 2, 7, 7),
                 'deformable_conv_offset': (2, 18, 3, 3),
                 'type_dict': {'deformable_conv_data': np.float32, 'deformable_conv_offset': np.float32}},
                {'ctx': mx.cpu(0),
                 'deformable_conv_data': (2, 2, 7, 7),
                 'deformable_conv_offset': (2, 18, 3, 3),
                 'type_dict': {'deformable_conv_data': np.float64, 'deformable_conv_offset': np.float64}},
                {'ctx': mx.cpu(0),
                 'deformable_conv_data': (2, 2, 7, 7),
                 'deformable_conv_offset': (2, 18, 3, 3),
                 'type_dict': {'deformable_conv_data': np.float32, 'deformable_conv_offset': np.float32}},
                ]
    sym = mx.sym.npx.deformable_convolution(num_filter=3, kernel=(3,3), stride=(2,2), name='deformable_conv')
    check_consistency(sym, ctx_list, scale=0.1, rtol=tol, atol=tol)

    # Dilate > 1
    ctx_list = [{'ctx': mx.gpu(0),
                 'deformable_conv_data': (2, 2, 7, 7),
                 'deformable_conv_offset': (2, 18, 3, 3),
                 'type_dict': {'deformable_conv_data': np.float64, 'deformable_conv_offset': np.float64}},
                {'ctx': mx.gpu(0),
                 'deformable_conv_data': (2, 2, 7, 7),
                 'deformable_conv_offset': (2, 18, 3, 3),
                 'type_dict': {'deformable_conv_data': np.float32, 'deformable_conv_offset': np.float32}},
                {'ctx': mx.cpu(0),
                 'deformable_conv_data': (2, 2, 7, 7),
                 'deformable_conv_offset': (2, 18, 3, 3),
                 'type_dict': {'deformable_conv_data': np.float64, 'deformable_conv_offset': np.float64}},
                {'ctx': mx.cpu(0),
                 'deformable_conv_data': (2, 2, 7, 7),
                 'deformable_conv_offset': (2, 18, 3, 3),
                 'type_dict': {'deformable_conv_data': np.float32, 'deformable_conv_offset': np.float32}},
                ]
    sym = mx.sym.npx.deformable_convolution(num_filter=3, kernel=(3,3), dilate=(2,2), name='deformable_conv')
    check_consistency(sym, ctx_list, scale=0.1, rtol=tol, atol=tol)

    # Deformable group > 1
    ctx_list = [{'ctx': mx.gpu(0),
                 'deformable_conv_data': (2, 2, 7, 7),
                 'deformable_conv_offset': (2, 36, 5, 5),
                 'type_dict': {'deformable_conv_data': np.float64, 'deformable_conv_offset': np.float64}},
                {'ctx': mx.gpu(0),
                 'deformable_conv_data': (2, 2, 7, 7),
                 'deformable_conv_offset': (2, 36, 5, 5),
                 'type_dict': {'deformable_conv_data': np.float32, 'deformable_conv_offset': np.float32}},
                {'ctx': mx.cpu(0),
                 'deformable_conv_data': (2, 2, 7, 7),
                 'deformable_conv_offset': (2, 36, 5, 5),
                 'type_dict': {'deformable_conv_data': np.float64, 'deformable_conv_offset': np.float64}},
                {'ctx': mx.cpu(0),
                 'deformable_conv_data': (2, 2, 7, 7),
                 'deformable_conv_offset': (2, 36, 5, 5),
                 'type_dict': {'deformable_conv_data': np.float32, 'deformable_conv_offset': np.float32}},
                ]
    sym = mx.sym.npx.deformable_convolution(num_filter=4, kernel=(3,3), num_deformable_group=2, name='deformable_conv')
    check_consistency(sym, ctx_list, scale=0.1, rtol=tol, atol=tol)


def check_rnn_layer(layer):
    layer.initialize(ctx=[mx.cpu(0), mx.gpu(0)])
    with mx.gpu(0):
        x = mx.nd.ones((10, 16, 30))
        states = layer.begin_state(16)
        go, gs = layer(x, states)

    with mx.cpu(0):
        x = mx.nd.ones((10, 16, 30))
        states = layer.begin_state(16)
        co, cs = layer(x, states)

    # atol of 1e-6 required, as exposed by seed 2124685726
    assert_almost_equal(go, co, rtol=1e-2, atol=1e-6)
    for g, c in zip(gs, cs):
        assert_almost_equal(g, c, rtol=1e-2, atol=1e-6)

def check_rnn_layer_w_rand_inputs(layer):
    layer.initialize(ctx=[mx.cpu(0), mx.gpu(0)])
    x = mx.nd.uniform(shape=(10, 16, 30))
    with mx.gpu(0):
        x = x.copyto(mx.gpu(0))
        states = layer.begin_state(16)
        go, gs = layer(x, states)

    with mx.cpu(0):
        x = x.copyto(mx.cpu(0))
        states = layer.begin_state(16)
        co, cs = layer(x, states)

    assert_almost_equal(go, co, rtol=1e-2, atol=1e-6)
    for g, c in zip(gs, cs):
        assert_almost_equal(g, c, rtol=1e-2, atol=1e-6)

@pytest.mark.serial
def test_sequence_reverse():
    check_sequence_reverse(mx.gpu(0))


@pytest.mark.serial
def test_autograd_save_memory():
    x = mx.nd.zeros((128, 512, 512), ctx=mx.gpu(0))
    x.attach_grad()

    with mx.autograd.record():
        for i in range(200):
            x = x + 1
            x.wait_to_read()
    x.backward()


@pytest.mark.serial
def test_cuda_rtc():
    source = r'''
    extern "C" __global__ void axpy(const float *x, float *y, float alpha) {
        int i = threadIdx.x + blockIdx.x * blockDim.x;
        y[i] += alpha * x[i];
    }

    extern "C" __global__ void saxpy(const float *x, float *y, float alpha) {
        extern __shared__ float smem[];
        int i = threadIdx.x + blockIdx.x * blockDim.x;
        smem[threadIdx.x] = x[i];
        y[i] += alpha * smem[threadIdx.x];
    }
    '''
    module = mx.rtc.CudaModule(source)
    axpy = module.get_kernel("axpy", "const float *x, float *y, float alpha")
    x = mx.nd.ones((10,), ctx=mx.gpu(0))
    y = mx.nd.zeros((10,), ctx=mx.gpu(0))
    axpy.launch([x, y, 3.0], mx.gpu(0), (1, 1, 1), (10, 1, 1))
    assert (y.asnumpy() == 3).all()

    saxpy = module.get_kernel("saxpy", "const float *x, float *y, float alpha")
    saxpy.launch([x, y, 4.0], mx.gpu(0), (1, 1, 1), (10, 1, 1), 10)
    assert (y.asnumpy() == 7).all()

    saxpy.launch([x, y, 5.0], mx.gpu(0), (2, 1, 1), (5, 1, 1), 5)
    assert (y.asnumpy() == 12).all()


@pytest.mark.serial
def test_cross_device_autograd():
    x = mx.nd.random.uniform(shape=(10,))
    x.attach_grad()

    with mx.autograd.record():
        y = mx.nd.tanh(x)
        y = y.copyto(mx.gpu(0))
        y = mx.nd.tanh(y)
        y = y.copyto(mx.cpu(0))
        y = mx.nd.tanh(y)
        y = y.copyto(mx.gpu(0))
        y = y.copyto(mx.gpu(0))

        y.backward()

    dx = x.grad.copy()
    x.grad[:] = 0

    with mx.autograd.record():
        y = x
        for i in range(3):
            y = mx.nd.tanh(y)
        y.backward()

    assert_almost_equal(dx, x.grad)

@pytest.mark.serial
def test_multi_proposal_op():
    # paramters
    feature_stride = 16
    scales = (8, 16, 32)
    ratios = (0.5, 1, 2)
    rpn_pre_nms_top_n = 12000
    rpn_post_nms_top_n = 2000
    rpn_min_size = feature_stride

    feat_len = (1000 + 15) // 16
    H, W = feat_len, feat_len
    num_anchors = len(scales) * len(ratios)
    count_anchors = H * W * num_anchors

    def get_new_data(batch_size, ctx):
        '''
        cls_prob: (batch_size, 2 * num_anchors, H, W)
        bbox_pred: (batch_size, 4 * num_anchors, H, W)
        im_info: (batch_size, 3)
        '''

        dtype = np.float32
        cls_prob = mx.nd.empty((batch_size, 2 * num_anchors, H, W), dtype = dtype, ctx = ctx)
        bbox_pred = mx.nd.empty((batch_size, 4 * num_anchors, H, W), dtype = dtype, ctx = ctx)
        im_info = mx.nd.empty((batch_size, 3), dtype = dtype, ctx = ctx)

        cls = [1.0 * (i + 1) / cls_prob.size for i in range(cls_prob.size)]
        np.random.shuffle(cls)
        cls_prob = mx.nd.reshape(mx.nd.array(cls, dtype = dtype, ctx = ctx), shape = cls_prob.shape)
        bbox_pred = mx.nd.array(np.random.randint(-2, 3, size = bbox_pred.shape), dtype = dtype, ctx = ctx)

        for i in range(batch_size):
            im_size = np.random.randint(600, feat_len * feature_stride, size = (2,))
            im_scale = np.random.randint(80, 100) / 100.0
            im_info[i, :] = [im_size[0], im_size[1], im_scale]
        return cls_prob, bbox_pred, im_info

    def check_proposal_consistency(op, batch_size, with_nms=False):
        '''
        op is mx.nd.contrib.Proposal or mx.nd.contrib.MultiProposal
        '''
        cls_prob, bbox_pred, im_info = get_new_data(batch_size, mx.cpu(0))
        rois_cpu, score_cpu = op(
                cls_prob = cls_prob,
                bbox_pred = bbox_pred,
                im_info = im_info,
                feature_stride = feature_stride,
                scales = scales,
                ratios = ratios,
                rpn_pre_nms_top_n = rpn_pre_nms_top_n,
                rpn_post_nms_top_n = rpn_post_nms_top_n,
                threshold = 0.7 if with_nms else 1.0,
                rpn_min_size = rpn_min_size, output_score = True)

        gpu_ctx = mx.gpu(0)

        # copy data to gpu from cpu
        cls_prob_gpu = cls_prob.as_in_context(gpu_ctx)
        bbox_pred_gpu = bbox_pred.as_in_context(gpu_ctx)
        im_info_gpu = im_info.as_in_context(gpu_ctx)

        rois_gpu, score_gpu = op(
                cls_prob = cls_prob_gpu,
                bbox_pred = bbox_pred_gpu,
                im_info = im_info_gpu,
                feature_stride = feature_stride,
                scales = scales,
                ratios = ratios,
                rpn_pre_nms_top_n = rpn_pre_nms_top_n,
                rpn_post_nms_top_n = rpn_post_nms_top_n,
                threshold = 0.7 if with_nms else 1.0,
                rpn_min_size = rpn_min_size, output_score = True)

        rois_cpu_np = rois_cpu.asnumpy()
        rois_gpu_np = rois_gpu.asnumpy()

        score_cpu_np = score_cpu.asnumpy()
        score_gpu_np = score_gpu.asnumpy()

        if not with_nms:
            assert_almost_equal(score_cpu_np, score_gpu_np, atol = 1e-3, rtol = 1e-3)
            assert_almost_equal(rois_cpu_np, rois_gpu_np, atol = 1e-3, rtol = 1e-3)
        else:
            # no 100% gurantee with nms
            assert(np.sum(np.abs(score_cpu_np - score_gpu_np) < 1e-3) >= 10)
            assert(np.sum(np.abs(rois_cpu_np - rois_gpu_np) < 1e-3) >= 40)

    check_proposal_consistency(mx.nd.contrib.Proposal, 1)
    check_proposal_consistency(mx.nd.contrib.MultiProposal, 5)
    check_proposal_consistency(mx.nd.contrib.Proposal, 1, with_nms=True)
    check_proposal_consistency(mx.nd.contrib.MultiProposal, 5, with_nms=True)


# The following 2 functions launch 0-thread kernels, an error that should be caught and signaled.
def kernel_error_check_imperative():
    with environment('MXNET_ENGINE_TYPE', 'NaiveEngine'):
        with mx.np_shape(active=True):
            a = mx.nd.array([1,2,3],ctx=mx.gpu(0))
            b = mx.nd.array([],ctx=mx.gpu(0))
            c = (a / b).asnumpy()

def kernel_error_check_symbolic():
    with environment('MXNET_ENGINE_TYPE', 'NaiveEngine'):
        with mx.np_shape(active=True):
            a = mx.sym.Variable('a')
            b = mx.sym.Variable('b')
            c = a / b
            f = c.bind(mx.gpu(0), {'a':mx.nd.array([1,2,3],ctx=mx.gpu(0)),
                                   'b':mx.nd.array([],ctx=mx.gpu(0))})
            f.forward()
            g = f.outputs[0].asnumpy()

@pytest.mark.serial
def test_kernel_error_checking():
    # Running tests that may throw exceptions out of worker threads will stop CI testing
    # if not run in a separate process (with its own address space for CUDA compatibility).
    try:
        mpctx = mp.get_context('spawn')
    except:
        print('SKIP: python%s.%s lacks the required process fork-exec support ... ' %
              sys.version_info[0:2], file=sys.stderr, end='')
    else:
        with discard_stderr():
            for f in [kernel_error_check_imperative, kernel_error_check_symbolic]:
                p = mpctx.Process(target=f)
                p.start()
                p.join()
                assert p.exitcode != 0,\
                    "Expected a synchronous kernel error from %s(), none seen." % f.__name__

def test_incorrect_gpu():
    # Try setting dev_id to a really big number
    pytest.raises(MXNetError, mx.nd.ones, (2,2), ctx=mx.gpu(100001))

def test_batchnorm_backwards_notrain():
    for ctx in [mx.cpu(0), mx.gpu(0)]:
        for cudnn_o in [False, True]:
            B,C,H,W = 4,3,2,2
            x = mx.nd.random.poisson(1,shape=(B,C,H,W)).as_in_context(ctx)
            gamma = mx.nd.random.normal(shape=(C)).as_in_context(ctx)
            beta = mx.nd.random.normal(shape=(C)).as_in_context(ctx)
            mean = mx.nd.random.normal(shape=(C)).as_in_context(ctx)
            std = mx.nd.random.normal(shape=(C)).as_in_context(ctx)
            x.attach_grad()

            with autograd.record(False):
                y = mx.ndarray.BatchNorm(x, gamma, beta, mean, std.square(),
                                         fix_gamma=False, cudnn_off=cudnn_o)
                loss=y.square().sum()
            loss.backward(train_mode=False)

def test_create_sparse_ndarray_gpu_to_cpu():
    dim0 = 10
    dim1 = 5
    densities = [0, 0.5, 1]
    for density in densities:
        shape = rand_shape_2d(dim0, dim1)
        matrix = rand_ndarray(shape, 'row_sparse', density)
        data = matrix.data
        indices = matrix.indices
        rsp_created = mx.nd.sparse.row_sparse_array((data, indices), shape=shape, ctx=mx.cpu())
        assert rsp_created.stype == 'row_sparse'
        assert same(rsp_created.data.asnumpy(), data.asnumpy())
        assert same(rsp_created.indices.asnumpy(), indices.asnumpy())
        rsp_copy = mx.nd.array(rsp_created)
        assert(same(rsp_copy.asnumpy(), rsp_created.asnumpy()))


def test_softmax_activation():
    gpu_a = mx.nd.array([[3., 0.5, -0.5, 2., 7.],
        [2., -.4, 7.,   3., 0.2]], ctx=mx.gpu(0))
    cpu_a = mx.nd.array([[3., 0.5, -0.5, 2., 7.],
        [2., -.4, 7.,   3., 0.2]], ctx=mx.cpu())

    cpu_a.attach_grad()
    gpu_a.attach_grad()
    with mx.autograd.record():
        gpu_y = mx.nd.SoftmaxActivation(data = gpu_a)
        cpu_y = mx.nd.SoftmaxActivation(data = cpu_a)
        assert_almost_equal(cpu_y, gpu_y, atol = 1e-3, rtol = 1e-3)

        gpu_y.backward()
        cpu_y.backward()
        assert_almost_equal(cpu_a.grad, gpu_a.grad, atol = 1e-3, rtol = 1e-3)


@pytest.mark.serial
@pytest.mark.serial
def test_bilinear_sampler_versions():
    data = mx.sym.Variable('data')
    grid = mx.sym.Variable('grid')
    sym1 = mx.sym.BilinearSampler(data=data, grid=grid)
    sym2 = mx.sym.BilinearSampler(data=data, grid=grid, cudnn_off=True)
    sym3 = mx.sym.BilinearSampler(data=data, grid=grid)

    test_cases = [[(1,3,15,16),(1,2,10,10)],
                 [(1,6,7,16),(1,2,10,4)],
                 [(1,7,3,16),(1,2,8,11)],
                 [(1,9,50,50),(1,2,50,50)]]

    for item in test_cases:
        data_shape, grid_shape = item
        # kWriteTo
        exe_cpu = sym1._simple_bind(data=data_shape, grid=grid_shape, ctx=mx.cpu(), grad_req='write')
        exe_gpu = sym2._simple_bind(data=data_shape, grid=grid_shape, ctx=default_context(), grad_req='write')
        exe_cudnn = sym3._simple_bind(data=data_shape, grid=grid_shape, ctx=default_context(), grad_req='write')
        exe_list = [exe_cpu, exe_gpu, exe_cudnn]
        ref_idx = 0
        test_data = np.random.uniform(low=-0.1, high=0.1,size=data_shape).astype(np.float32)
        test_grid = np.random.uniform(low=-2, high=2, size=grid_shape).astype(np.float32)
        for exe in exe_list:
            exe.arg_dict['data'][:] = test_data
            exe.arg_dict['grid'][:] = test_grid
            exe.forward(is_train=True)
            mx.test_utils.assert_almost_equal(exe_list[ref_idx].outputs[0], exe.outputs[0], rtol=1e-3, atol=1e-5)

        out_grad = np.random.uniform(low=-0.01, high=0.01,size=data_shape[:2] + grid_shape[2:]).astype(np.float32)
        for exe in exe_list:
            exe.backward(mx.nd.array(out_grad))
            assert_almost_equal(exe.grad_dict['data'], exe_list[ref_idx].grad_dict['data'], rtol=1e-3, atol=1e-5)
            assert_almost_equal(exe.grad_dict['grid'], exe_list[ref_idx].grad_dict['grid'], rtol=1e-3, atol=1e-5)

        data_grad = exe_list[ref_idx].grad_dict['data'].asnumpy()
        grid_grad = exe_list[ref_idx].grad_dict['grid'].asnumpy()

        # kAddTo
        exe_cpu_addto = sym1._simple_bind(data=data_shape, grid=grid_shape, ctx=mx.cpu(), grad_req='add')
        exe_gpu_addto = sym2._simple_bind(data=data_shape, grid=grid_shape, ctx=default_context(), grad_req='add')
        exe_cudnn_addto = sym3._simple_bind(data=data_shape, grid=grid_shape, ctx=default_context(), grad_req='add')
        exe_list = [exe_cpu_addto, exe_gpu_addto, exe_cudnn_addto]
        data_initial_grad = np.random.normal(size=exe_list[ref_idx].grad_dict['data'].shape).astype(np.float32)
        grid_initial_grad = np.random.normal(size=exe_list[ref_idx].grad_dict['grid'].shape).astype(np.float32)
        for exe in exe_list:
            exe.arg_dict['data'][:] = test_data
            exe.arg_dict['grid'][:] = test_grid
            exe.grad_dict['data'][:] = data_initial_grad
            exe.grad_dict['grid'][:] = grid_initial_grad
            exe.forward(is_train=True)
            exe.backward(mx.nd.array(out_grad))
            assert_almost_equal(exe.grad_dict['data'], exe_list[ref_idx].grad_dict['data'], rtol=1e-3, atol=1e-5)
            assert_almost_equal(exe.grad_dict['grid'], exe_list[ref_idx].grad_dict['grid'], rtol=1e-3, atol=1e-5)
        assert_almost_equal(exe_list[ref_idx].grad_dict['data'], data_grad + data_initial_grad, rtol=1e-3, atol=1e-5)
        assert_almost_equal(exe_list[ref_idx].grad_dict['grid'], grid_grad + grid_initial_grad, rtol=1e-3, atol=1e-5)

        for req_dict in [{'data' : 'null', 'grid' : 'write'}, {'data' : 'write', 'grid' : 'null'}]:
            # Mixture of kWriteTo and kNullOp
            exe_cpu_mix = sym1._simple_bind(data=data_shape, grid=grid_shape, ctx=mx.cpu(), grad_req=req_dict)
            exe_gpu_mix = sym2._simple_bind(data=data_shape, grid=grid_shape, ctx=default_context(), grad_req=req_dict)
            exe_cudnn_mix = sym3._simple_bind(data=data_shape, grid=grid_shape, ctx=default_context(), grad_req=req_dict)
            exe_list = [exe_cpu_mix, exe_gpu_mix, exe_cudnn_mix]
            for exe in exe_list:
                exe.arg_dict['data'][:] = test_data
                exe.arg_dict['grid'][:] = test_grid
                exe.forward(is_train=True)
                exe.backward(mx.nd.array(out_grad))
                if req_dict['data'] is 'write':
                    assert_almost_equal(exe.grad_dict['data'], exe_list[ref_idx].grad_dict['data'], rtol=1e-3, atol=1e-5)
                if req_dict['grid'] is 'write':
                    assert_almost_equal(exe.grad_dict['grid'], exe_list[ref_idx].grad_dict['grid'], rtol=1e-3, atol=1e-5)


# isolated execution bulking test function to be invoked with different env var settings
def _test_bulking_in_process(seed, time_per_iteration):
    data_shape = (10,)
    num_ops = 1000
    num_iterations = 20

    ctx = default_context()
    # build symbol
    X = mx.sym.Variable('X')
    sym = mx.sym.flip(X, axis=0)
    for _ in range(num_ops-1):
        sym = mx.sym.flip(sym, axis=0)
    x = mx.ndarray.zeros(data_shape)
    dx = mx.ndarray.zeros(data_shape)
    dy = mx.ndarray.ones(data_shape)
    exe = sym._bind(ctx=ctx, args=[x], args_grad = {'X':dx})

    # time a number of forward() and backward() executions after some warm-up iterations
    warmups = 1
    for i in range(num_iterations+warmups):
        if i == warmups:
            start = time.time()
        exe.forward(is_train=True)
        exe.backward(dy)
        dx.wait_to_read()
    time_per_iteration.value = (time.time() - start) / num_iterations


@pytest.mark.skip(reason='skippping temporarily, tracked by https://github.com/apache/incubator-mxnet/issues/16517')
def test_bulking_operator_gpu():
    _test_bulking(_test_bulking_in_process)


@pytest.mark.skip(reason='skippping temporarily, tracked by https://github.com/apache/incubator-mxnet/issues/14970')
def test_bulking():
    # test case format: (max_fwd_segment_size, max_bwd_segment_size, enable_bulking_in_training)
    test_cases = [(0,0,True), (1,1,True), (15,15,False), (15,0,True), (0,15,True), (15,15,True)]
    times = {}
    times_str = ''
    for seg_sizes in test_cases:
        # Create shared variable to return measured time from test process
        time_per_iteration = mp.Manager().Value('d', 0.0)
        if not run_in_spawned_process(_test_bulking_in_process,
                                      {'MXNET_EXEC_BULK_EXEC_MAX_NODE_TRAIN_FWD' : str(seg_sizes[0]),
                                       'MXNET_EXEC_BULK_EXEC_MAX_NODE_TRAIN_BWD' : str(seg_sizes[1]),
                                       'MXNET_EXEC_BULK_EXEC_TRAIN' : str(seg_sizes[2])},
                                      time_per_iteration):
            # skip test since the python version can't run it properly.  Warning msg was logged.
            return
        times[seg_sizes] = time_per_iteration.value
        times_str += \
            '\n    runtime of (fwd,bwd,enable) op seg setting ({},{},{}) =\t{:.1f} msec'.format(
            seg_sizes[0], seg_sizes[1], seg_sizes[2], 1000.0 * times[seg_sizes])

    fastest_non_bulked_time = min(times[(0,0,True)], times[(1,1,True)], times[(15,15,False)])
    slowest_half_bulked_time = max(times[(0,15,True)], times[(15,0,True)])
    fastest_half_bulked_time = min(times[(0,15,True)], times[(15,0,True)])
    fully_bulked_time = times[(15,15,True)]

    print(times_str)
    # Non-bulked times[0,0,True], times[1,1,True] and times[15,15,False] should be about the same,
    # slower than both half-bulked times[0,15,True] and times[15,0,True]
    assert slowest_half_bulked_time < fastest_non_bulked_time, \
        'A half-bulked exec time is slower than the non-bulked time by {} secs! {}' \
            .format(slowest_half_bulked_time - fastest_non_bulked_time, times_str)
    # The fully bulked times[15,15,True] should be faster than both half-bulked runs
    assert fully_bulked_time < fastest_half_bulked_time, \
        'The fully-bulked exec time is slower than a half-bulked time by {} secs! {}' \
            .format(fully_bulked_time - fastest_half_bulked_time, times_str)


@pytest.mark.serial
def test_allclose_function_gpu():
    allclose_function([mx.cpu(), mx.gpu(0)])

def test_context_num_gpus():
    # Test that num_gpus reports at least one GPU, as the test is run on a GPU host.
    assert mx.context.num_gpus() > 0

def math_log(shape, dtype, check_value):
    np_x = np.random.rand(*tuple(shape))
    x = mx.nd.array(np_x, dtype=dtype)
    y = mx.nd.log(data=x)
    if check_value:
        x_ = x.as_in_context(mx.cpu())
        y_ = mx.nd.log(data=x_)
        assert_almost_equal(y.asnumpy(), y_.asnumpy())

def math_erf(shape, dtype, check_value):
    np_x = np.random.rand(*tuple(shape))
    x = mx.nd.array(np_x, dtype=dtype)
    y = mx.nd.erf(data=x)
    if check_value:
        x_ = x.as_in_context(mx.cpu())
        y_ = mx.nd.erf(data=x_)
        assert_almost_equal(y.asnumpy(), y_.asnumpy())

def math_square(shape, dtype, check_value):
    np_x = np.random.rand(*tuple(shape))
    x = mx.nd.array(np_x, dtype=dtype)
    y = mx.nd.square(data=x)
    if check_value:
        x_ = x.as_in_context(mx.cpu())
        y_ = mx.nd.square(data=x_)
        assert_almost_equal(y.asnumpy(), y_.asnumpy())

def run_math(op, shape, dtype="float32", check_value=True):
    run_num = 10
    for i in range(run_num):
        if op == 'log':
            math_log(shape=shape, dtype=dtype, check_value=check_value)
        elif op == 'erf':
            math_erf(shape=shape, dtype=dtype, check_value=check_value)
        elif op == 'square':
            math_square(shape=shape, dtype=dtype, check_value=check_value)

@pytest.mark.serial
def test_math():
    ops = ['log', 'erf', 'square']
    check_value= True
    shape_lst = [[1000], [100,1000], [10,100,100], [10,100,100,100]]
    dtypes = ["float32", "float64"]
    for shape in shape_lst:
        for dtype in dtypes:
            for op in ops:
                run_math(op, shape, dtype, check_value=check_value)

@pytest.mark.serial
def test_arange_like_dtype():
    dtypes = [np.float16, np.float32, np.float64]

    for t in dtypes:
        x = mx.sym.Variable('x', dtype=t)
        y = mx.sym.reshape(x, shape=(0, 0, -1))
        z = mx.sym.contrib.arange_like(y, axis=-1)

        mod = z._simple_bind(ctx=mx.gpu(0), x=(3, 4, 5, 6), grad_req='null')
        mod.arg_arrays[0][:] = np.random.normal(size=mod.arg_arrays[0].shape).astype(t)
        out = mod.forward(is_train=False)
        for v in out:
            assert v.dtype == t


def test_fp16_spmm():
    inp = mxsps.csr_matrix(sps.coo_matrix(([2.0], ([150], [100000]))).tocsr())
    inp = inp.astype('float16', copy=False)
    weight = mx.nd.random.randn(100001, 151)
    weight = weight.astype('float16', copy=False)
    out = mxsps.dot(inp, weight)
    out_np = mx.nd.dot(inp, weight)
    assert_almost_equal(out.asnumpy(), out_np, rtol=1e-3, atol=1e-5)

@pytest.mark.serial
@pytest.mark.parametrize('dtype', ["float16", "float32", "float64"])
def test_split_v2_fwd(dtype):
    dim = random.randint(2, 9)
    shape = rand_shape_nd(dim)
    axis = random.randint(-dim, dim-1)
    axis_size = shape[axis]
    samples = random.randint(0, axis_size - 1)
    indices = sorted(random.sample([i for i in range(1, axis_size)], samples))
    indices = tuple(indices)
    mx_data = rand_ndarray(shape, dtype=dtype)
    np_data = mx_data.asnumpy()
    np_out = np.split(np_data, indices_or_sections=indices, axis=axis)
    data = mx.sym.Variable("data")
    sym = mx.sym.split_v2(data, indices_or_sections=indices, axis=axis)
    check_symbolic_forward(sym, {"data": mx_data}, np_out, rtol=1e-3, atol=1e-5)
