import mxnet as mx
import numpy as np
import argparse
import ctypes
import unittest
from common import with_seed
from mxnet.io import NDArrayIter
from mxnet.module import Module
from mxnet.symbol import Symbol
from importlib import import_module
from numpy.testing import assert_allclose
from mxnet.base import SymbolHandle, check_call, _LIB, mx_uint, c_str
from mxnet.test_utils import DummyIter

def check_qsym_calibrated(qsym):
  attrs = qsym.attr_dict()
  min_value = 0.0
  max_value = 0.0
  assert ''.join(qsym.attr_dict().keys()).find('quantized_') != -1
  for k, v in attrs.items():
    if k.find('quantized_sg_mkldnn_conv') != -1:
      assert 'min_calib_range' in v
      assert 'max_calib_range' in v
      min_value = v['min_calib_range']
      max_value = v['max_calib_range']
    if k.find('_quantize') != -1:
      assert v['out_type'] == 'uint8'
  return float(min_value), float(max_value)

def check_qsym_forward(qsym, qarg_params, qaux_params, data_val, data_shape, label_shape):
  mod = mx.mod.Module(symbol=qsym, context=mx.current_context())
  mod.bind(for_training=False,
           data_shapes=[('data', data_shape)],
           label_shapes=[('softmax_label', label_shape)])
  mod.set_params(qarg_params, qaux_params)
  batch = mx.io.DataBatch(data_val, [])
  mod.forward(batch, is_train=False)
  for output in mod.get_outputs():
    output.wait_to_read()
  return output

def check_quantize(sym, data_shape, label_shape, data_val, sym_output):
    mod = Module(symbol=sym)
    mod.bind(data_shapes=[('data', data_shape)], label_shapes=[('softmax_label', label_shape)], for_training=False)
    mod.init_params()
    arg_params, aux_params = mod.get_params()
    excluded_sym_names = []
    if mx.current_context() == mx.cpu():
      excluded_sym_names += ['fc']
    calib_data = mx.nd.random.uniform(shape=data_shape)
    calib_data = NDArrayIter(data=calib_data)
    calib_data = DummyIter(calib_data)
    calib_layer = lambda name: name.endswith('_output')
    qsym, qarg_params, qaux_params = mx.contrib.quant.quantize_model(sym=sym,
                                                                     arg_params=arg_params,
                                                                     aux_params=aux_params,
                                                                     ctx=mx.current_context(),
                                                                     excluded_sym_names=excluded_sym_names,
                                                                     quantized_dtype='uint8',
                                                                     calib_mode='naive',
                                                                     calib_data=calib_data,
                                                                     calib_layer=calib_layer,
                                                                     disable_requantize=True,
                                                                     calib_quantize_op=True,
                                                                     num_calib_examples=20)
    minVar, maxVar = check_qsym_calibrated(qsym)
    rtol = (maxVar - minVar) / 256
    qsym_output = check_qsym_forward(qsym, qarg_params, qaux_params, data_val, data_shape, label_shape)
    assert_allclose(qsym_output[0].asnumpy(), sym_output[0].asnumpy(), rtol=rtol)

def check_fusion(sym, name, nofusion=False):
  shape = (4, 4, 10, 10)
  label_shape = (4, 10)
  exe = sym.simple_bind(mx.cpu(), data=shape, grad_req='null')
  out = SymbolHandle()
  backend = "MKLDNN"
  check_call(_LIB.MXGenBackendSubgraph(c_str(backend), sym.handle, ctypes.byref(out)))
  sym_sg = Symbol(out)
  exe_sg = sym_sg.simple_bind(mx.cpu(), data=shape, grad_req='null')

  for k, v in exe.arg_dict.items():
    v = mx.random.uniform(-1.0, 1.0, shape=v.shape)
  data_val = [exe.arg_dict['data']]

  fwd = exe.forward(is_train=False)
  fwd[0].wait_to_read()

  fwd_sg = exe_sg.forward(is_train=False)
  fwd_sg[0].wait_to_read()

  # Check the result accuracy based on fp32 fusion
  assert_allclose(fwd[0].asnumpy(), fwd_sg[0].asnumpy(), rtol=0)
  attrs=sym_sg.attr_dict()
  if not nofusion:
    assert ''.join(sym_sg.get_internals().list_outputs()).find('sg_mkldnn_conv') != -1
  for k, v in attrs.items():
    if k.find('sg_mkldnn_conv') != -1:
      for attr_op in name:
        assert v[attr_op] == 'true'

  # fp32 to int8
  if nofusion:
    check_quantize(sym, shape, label_shape, data_val, fwd[0])
  else: check_quantize(sym_sg, shape, label_shape, data_val, fwd[0])

def single_conv():
  data = mx.symbol.Variable('data')
  weight = mx.symbol.Variable('weight')
  bn = mx.sym.BatchNorm(data=data, fix_gamma=True, eps=2e-5, momentum=0.9, name='bn')
  conv = mx.symbol.Convolution(data=bn, weight=weight, name='conv', num_filter=64, kernel=(3, 3), stride=(1, 1))
  fc = mx.sym.FullyConnected(data=conv, num_hidden=10, flatten=True, name='fc')
  sym = mx.sym.SoftmaxOutput(data=fc, name='softmax')
  return sym

def conv_bn():
  data = mx.symbol.Variable('data')
  weight = mx.symbol.Variable('weight')
  bn1 = mx.sym.BatchNorm(data=data, fix_gamma=True, eps=2e-5, momentum=0.9, name='bn1')
  conv = mx.symbol.Convolution(data=bn1, weight=weight, name='conv', num_filter=64, kernel=(3, 3), stride=(1, 1))
  bn = mx.symbol.BatchNorm(data=conv, name="bn")
  fc = mx.sym.FullyConnected(data=bn, num_hidden=10, flatten=True, name='fc')
  sym = mx.sym.SoftmaxOutput(data=fc, name='softmax')
  return sym

def conv_relu():
  data = mx.symbol.Variable('data')
  weight = mx.symbol.Variable('weight')
  bn = mx.sym.BatchNorm(data=data, fix_gamma=True, eps=2e-5, momentum=0.9, name='bn')
  conv = mx.symbol.Convolution(data=bn, weight=weight, name='conv', num_filter=64, kernel=(3, 3), stride=(1, 1))
  relu = mx.symbol.Activation(data=conv, name='relu', act_type="relu")
  fc = mx.sym.FullyConnected(data=relu, num_hidden=10, flatten=True, name='fc')
  sym = mx.sym.SoftmaxOutput(data=fc, name='softmax')
  return sym

def conv_sum():
  data = mx.symbol.Variable('data')
  weight = mx.symbol.Variable('weight')
  bn = mx.sym.BatchNorm(data=data, fix_gamma=True, eps=2e-5, momentum=0.9, name='bn')
  conv = mx.symbol.Convolution(data=bn, weight=weight, name='conv', num_filter=64, kernel=(3, 3), stride=(1, 1))
  conv1 = mx.symbol.Convolution(data=bn, weight=weight, name='conv1', num_filter=64, kernel=(3, 3), stride=(1, 1))
  sum1 = conv + conv1
  fc = mx.sym.FullyConnected(data=sum1, num_hidden=10, flatten=True, name='fc')
  sym = mx.sym.SoftmaxOutput(data=fc, name='softmax')
  return sym

def conv_bn_relu():
  data = mx.symbol.Variable('data')
  weight = mx.symbol.Variable('weight')
  bn1 = mx.sym.BatchNorm(data=data, fix_gamma=True, eps=2e-5, momentum=0.9, name='bn1')
  conv = mx.symbol.Convolution(data=bn1, weight=weight, name='conv', num_filter=64, kernel=(3, 3), stride=(1, 1))
  bn = mx.symbol.BatchNorm(data=conv, name="bn")
  relu = mx.symbol.Activation(data=bn, name='relu', act_type="relu")
  fc = mx.sym.FullyConnected(data=relu, num_hidden=10, flatten=True, name='fc')
  sym = mx.sym.SoftmaxOutput(data=fc, name='softmax')
  return sym

def conv_bn_sum_relu():
  data = mx.symbol.Variable('data')
  weight = mx.symbol.Variable('weight')
  bn1 = mx.symbol.BatchNorm(data=data, name="bn1")
  conv = mx.symbol.Convolution(data=bn1, weight=weight, name='conv', num_filter=64, kernel=(3, 3), stride=(1, 1))
  bn = mx.symbol.BatchNorm(data=conv, name="bn")
  conv1 = mx.symbol.Convolution(data=bn1, weight=weight, name='conv1', num_filter=64, kernel=(3, 3), stride=(1, 1))
  sum1 = bn + conv1
  relu = mx.symbol.Activation(data=sum1, name='relu', act_type="relu")
  fc = mx.sym.FullyConnected(data=relu, num_hidden=10, flatten=True, name='fc')
  sym = mx.sym.SoftmaxOutput(data=fc, name='softmax')
  return sym

def int8_pooling():
  data = mx.symbol.Variable('data')
  bn = mx.sym.BatchNorm(data=data, fix_gamma=True, eps=2e-5, momentum=0.9, name='bn')
  pool = mx.sym.Pooling(data=bn, kernel=(4, 4), pool_type='avg', name='pool')
  fc = mx.sym.FullyConnected(data=pool, num_hidden=10, flatten=True, name='fc')
  sym = mx.sym.SoftmaxOutput(data=fc, name='softmax')
  return sym

@with_seed()
def test_sugbraph():
  conv_op = ['']
  conv_relu_op = ['with_relu']
  conv_bn_op = ['with_bn']
  conv_sum_op = ['with_sum']
  conv_bn_relu_op = ['with_bn', 'with_relu']
  conv_bn_sum_relu_op = ['with_sum', 'with_postsum_relu', 'with_bn']

  net = conv_bn_sum_relu()
  check_fusion(net, conv_bn_sum_relu_op)
  net = single_conv()
  check_fusion(net, conv_op)
  net = conv_relu()
  check_fusion(net, conv_relu_op)
  net = conv_bn()
  check_fusion(net, conv_bn_op)
  net = conv_sum()
  check_fusion(net, conv_sum_op)
  net = conv_bn_relu()
  check_fusion(net, conv_bn_relu_op)
  net = int8_pooling()
  check_fusion(net, conv_op, True)