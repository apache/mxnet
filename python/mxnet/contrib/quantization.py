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
"""Quantization module for generating quantized (INT8) models from FP32 models."""


try:
    from scipy import stats
except ImportError:
    stats = None

import ctypes
import logging
import os
import shutil
import warnings
import numpy as np
from ..base import _LIB, check_call, py_str
from ..base import c_array, c_str, mx_uint, c_str_array
from ..base import NDArrayHandle, SymbolHandle
from ..symbol import Symbol
from ..symbol import load as sym_load
from .. import ndarray
from ..ndarray import load as nd_load
from ..ndarray import save as nd_save
from ..ndarray import NDArray
from ..io import DataIter, DataDesc, DataBatch
from ..context import cpu, Context
from ..module import Module


def _quantize_params(qsym, params, th_dict):
    """Given a quantized symbol and a dict of params that have not been quantized,
    generate quantized params. Currently only supports quantizing the arg_params
    with names of `weight` or `bias`, not aux_params. If `qsym` contains symbols
    that are excluded from being quantized, their corresponding params will
    not be quantized, but saved together with quantized params of the symbols that
    have been quantized.

    Parameters
    ----------
    qsym : Symbol
        Quantized symbol from FP32 symbol.
    params : dict of str->NDArray
    th_dict: dict of min/max pairs of layers' output
    """
    inputs_name = qsym.list_arguments()
    quantized_params = {}
    for name in inputs_name:
        if name.endswith(('weight_quantize', 'bias_quantize')):
            original_name = name[:-len('_quantize')]
            param = params[original_name]
            # pylint: disable=unbalanced-tuple-unpacking
            val, vmin, vmax = ndarray.contrib.quantize(data=param,
                                                       min_range=ndarray.min(param),
                                                       max_range=ndarray.max(param),
                                                       out_type='int8')
            quantized_params[name] = val
            quantized_params[name+'_min'] = vmin
            quantized_params[name+'_max'] = vmax
        elif name in params:
            quantized_params[name] = params[name]
        elif name.endswith(('_min')):
            output = name[: - len('_min')]
            if output in th_dict:
                quantized_params[name] = ndarray.array([th_dict[output][0]])
        elif name.endswith(('_max')):
            output = name[: - len('_min')]
            if output in th_dict:
                quantized_params[name] = ndarray.array([th_dict[output][1]])
    return quantized_params

def _quantize_symbol(sym, ctx, excluded_symbols=None, excluded_operators=None,
                     offline_params=None, quantized_dtype='int8', quantize_mode='smart',
                     quantize_granularity='tensor-wise'):
    """Given a symbol object representing a neural network of data type FP32,
    quantize it into a INT8 network.

    Parameters
    ----------
    sym : Symbol
        FP32 neural network symbol.
    ctx : Context
        Defines the device that users want to run quantized symbol.
    excluded_symbols : list of strings
        A list of strings representing the names of the symbols that users want to excluding
        from being quantized.
    excluded_operators : list of strings
        A list of strings representing the names of the operators that users want to excluding
        from being quantized.
    offline_params : list of strs
        Names of the parameters that users want to quantize offline. It's always recommended to
        quantize parameters offline so that quantizing parameters during the inference can be
        avoided.
    quantized_dtype: str
        The quantized destination type for input data.
    quantize_mode: str
        The mode that quantization pass to apply.
    quantize_granularity: str
        The granularity of quantization, currently supports 'tensor-wise' and 'channel-wise'
        quantization. The default value is 'tensor-wise'.

    """
    num_excluded_symbols = 0
    if excluded_symbols is not None:
        assert isinstance(excluded_symbols, list)
        num_excluded_symbols = len(excluded_symbols)
    else:
        excluded_symbols = []

    num_excluded_ops = 0
    if excluded_operators is not None:
        assert isinstance(excluded_operators, list)
        num_excluded_ops = len(excluded_operators)
    else:
        excluded_operators = []

    num_offline = 0
    offline = []
    if offline_params is not None:
        num_offline = len(offline_params)
        for k in offline_params:
            offline.append(c_str(k))

    out = SymbolHandle()
    size = mx_uint()
    calib_str = ctypes.POINTER(ctypes.c_char_p)()
    check_call(_LIB.MXQuantizeSymbol(sym.handle,
                                     ctypes.byref(out),
                                     ctypes.byref(ctypes.c_int(ctx.device_typeid)),
                                     mx_uint(num_excluded_symbols),
                                     c_str_array(excluded_symbols),
                                     mx_uint(num_excluded_ops),
                                     c_str_array(excluded_operators),
                                     mx_uint(num_offline),
                                     c_array(ctypes.c_char_p, offline),
                                     c_str(quantized_dtype),
                                     ctypes.c_bool(True),
                                     c_str(quantize_mode),
                                     c_str(quantize_granularity),
                                     ctypes.byref(size),
                                     ctypes.byref(calib_str)))
    calib_layer = []
    calib_layer = [py_str(calib_str[i]) for i in range(size.value)]
    return Symbol(out), calib_layer

def combine_histogram(old_hist, arr, new_min, new_max, new_th):
    """ Collect layer histogram for arr and combine it with old histogram.
    """
    (old_hist, old_hist_edges, old_min, old_max, old_th) = old_hist
    if new_th <= old_th:
        hist, _ = np.histogram(arr, bins=len(old_hist), range=(-old_th, old_th))
        return (old_hist + hist, old_hist_edges, min(old_min, new_min), max(old_max, new_max), old_th)
    else:
        # Need to generate new histogram with new_th
        old_num_bins = len(old_hist)
        old_step = 2 * old_th / old_num_bins
        half_increased_bins = int((new_th - old_th) // old_step + 1)
        new_num_bins = half_increased_bins * 2 + old_num_bins
        new_th = half_increased_bins * old_step + old_th
        hist, hist_edges = np.histogram(arr, bins=new_num_bins, range=(-new_th, new_th))
        hist[half_increased_bins:new_num_bins - half_increased_bins] += old_hist
        return (hist, hist_edges, min(old_min, new_min), max(old_max, new_max), new_th)

class _LayerHistogramCollector(object):
    """Saves layer histogram in a dict with layer names as keys and lists of NDArrays as
    values. The collected histogram will be used for calculating the optimal thresholds for
    quantization using KL divergence.
    """
    def __init__(self, num_bins=8001, include_layer=None, logger=None):
        self.hist_dict = {}
        self.num_bins = num_bins
        self.include_layer = include_layer
        self.logger = logger

    def collect(self, name, arr):
        """Callback function for collecting layer output NDArrays."""
        name = py_str(name)
        if name not in self.include_layer:
            return
        handle = ctypes.cast(arr, NDArrayHandle)
        arr = NDArray(handle, writable=False).copyto(cpu()).asnumpy()
        if self.logger:
            self.logger.debug("Collecting layer %s histogram of shape %s" % (name, arr.shape))
        min_range = np.min(arr)
        max_range = np.max(arr)
        th = max(abs(min_range), abs(max_range))
        if name in self.hist_dict:
            self.hist_dict[name] = combine_histogram(self.hist_dict[name], arr, min_range, max_range, th)
        else:
            hist, hist_edges = np.histogram(arr, bins=self.num_bins, range=(-th, th))
            self.hist_dict[name] = (hist, hist_edges, min_range, max_range, th)

class _LayerOutputMinMaxCollector(object):
    """Saves layer output min and max values in a dict with layer names as keys.
    The collected min and max values will be directly used as thresholds for quantization.
    """
    def __init__(self, quantized_dtype, include_layer=None, logger=None):
        self.min_max_dict = {}
        self.quantized_dtype = quantized_dtype
        self.include_layer = include_layer
        self.logger = logger

    def collect(self, name, arr):
        """Callback function for collecting min and max values from an NDArray."""
        name = py_str(name)
        if name not in self.include_layer:
            return
        handle = ctypes.cast(arr, NDArrayHandle)
        arr = NDArray(handle, writable=False)
        min_range = ndarray.min(arr).asscalar()
        max_range = ndarray.max(arr).asscalar()
        if name in self.min_max_dict:
            cur_min_max = self.min_max_dict[name]
            self.min_max_dict[name] = (min(cur_min_max[0], min_range),
                                       max(cur_min_max[1], max_range))
        else:
            self.min_max_dict[name] = (min_range, max_range)
        if self.logger:
            self.logger.debug("Collecting layer %s min_range=%f, max_range=%f"
                              % (name, min_range, max_range))

def _calibrate_quantized_sym(qsym, th_dict):
    """Given a dictionary containing the thresholds for quantizing the layers,
    set the thresholds into the quantized symbol as the params of requantize operators.
    """
    if th_dict is None or len(th_dict) == 0:
        return qsym
    num_layer_outputs = len(th_dict)
    layer_output_names = []
    min_vals = []
    max_vals = []
    for k, v in th_dict.items():
        layer_output_names.append(k)
        min_vals.append(v[0])
        max_vals.append(v[1])

    calibrated_sym = SymbolHandle()
    check_call(_LIB.MXSetCalibTableToQuantizedSymbol(qsym.handle,
                                                     mx_uint(num_layer_outputs),
                                                     c_str_array(layer_output_names),
                                                     c_array(ctypes.c_float, min_vals),
                                                     c_array(ctypes.c_float, max_vals),
                                                     ctypes.byref(calibrated_sym)))
    return Symbol(calibrated_sym)


def _collect_layer_statistics(mod, data, collector, max_num_examples=None, logger=None):
    if not isinstance(data, DataIter):
        raise ValueError('Only supports data as a type of DataIter, while received type %s'
                         % str(type(data)))
    mod._exec_group.execs[0].set_monitor_callback(collector.collect, monitor_all=True)
    num_batches = 0
    num_examples = 0
    for batch in data:
        mod.forward(data_batch=batch, is_train=False)
        num_batches += 1
        num_examples += data.batch_size
        if max_num_examples is not None and num_examples >= max_num_examples:
            break
    if logger is not None:
        logger.info("Collected statistics from %d batches with batch_size=%d"
                    % (num_batches, data.batch_size))
    return num_examples


def _collect_layer_output_min_max(mod, data, quantized_dtype, include_layer=None,
                                  max_num_examples=None, logger=None):
    """Collect min and max values from layer outputs and save them in
    a dictionary mapped by layer names.
    """
    collector = _LayerOutputMinMaxCollector(quantized_dtype=quantized_dtype,
                                            include_layer=include_layer, logger=logger)
    num_examples = _collect_layer_statistics(mod, data, collector, max_num_examples, logger)
    return collector.min_max_dict, num_examples


def _collect_layer_histogram(mod, data, include_layer=None,
                             max_num_examples=None, logger=None):
    """Collect layer outputs and save them in a dictionary mapped by layer names."""
    collector = _LayerHistogramCollector(include_layer=include_layer, logger=logger)
    num_examples = _collect_layer_statistics(mod, data, collector, max_num_examples, logger)
    return collector.hist_dict, num_examples


def _smooth_distribution(p, eps=0.0001):
    """Given a discrete distribution (may have not been normalized to 1),
    smooth it by replacing zeros with eps multiplied by a scaling factor and taking the
    corresponding amount off the non-zero values.
    Ref: http://web.engr.illinois.edu/~hanj/cs412/bk3/KL-divergence.pdf
    """
    is_zeros = (p == 0).astype(np.float32)
    is_nonzeros = (p != 0).astype(np.float32)
    n_zeros = is_zeros.sum()
    n_nonzeros = p.size - n_zeros
    if not n_nonzeros:
        raise ValueError('The discrete probability distribution is malformed. All entries are 0.')
    eps1 = eps * float(n_zeros) / float(n_nonzeros)
    assert eps1 < 1.0, 'n_zeros=%d, n_nonzeros=%d, eps1=%f' % (n_zeros, n_nonzeros, eps1)
    hist = p.astype(np.float32)
    hist += eps * is_zeros + (-eps1) * is_nonzeros
    assert (hist <= 0).sum() == 0
    return hist


# pylint: disable=line-too-long
def _get_optimal_threshold(hist_data, quantized_dtype, num_quantized_bins=255):
    """Given a dataset, find the optimal threshold for quantizing it.
    The reference distribution is `q`, and the candidate distribution is `p`.
    `q` is a truncated version of the original distribution.

    Ref: http://on-demand.gputechconf.com/gtc/2017/presentation/s7310-8-bit-inference-with-tensorrt.pdf
    """
    (hist, hist_edges, min_val, max_val, _) = hist_data
    num_bins = len(hist)
    assert (num_bins % 2 == 1)
    if min_val >= 0 and quantized_dtype in ['auto', 'uint8']:
        # We need to move negative bins to positive bins to fit uint8 range.
        num_quantized_bins = num_quantized_bins * 2 + 1
    hist = ndarray.array(hist, ctx=cpu())
    hist_edges = ndarray.array(hist_edges, ctx=cpu())
    threshold, divergence = ndarray.contrib.calibrate_entropy(hist=hist,
                                                              hist_edges=hist_edges,
                                                              num_quantized_bins=num_quantized_bins)
    threshold = threshold.asnumpy()
    divergence = divergence.asnumpy()
    return min_val, max_val, threshold, divergence
# pylint: enable=line-too-long

def _get_optimal_thresholds(hist_dict, quantized_dtype, num_quantized_bins=255, logger=None):
    """Given a ndarray dict, find the optimal threshold for quantizing each value of the key."""
    if stats is None:
        raise ImportError('scipy.stats is required for running entropy mode of calculating'
                          ' the optimal thresholds for quantizing FP32 ndarrays into int8.'
                          ' Please check if the scipy python bindings are installed.')
    assert isinstance(hist_dict, dict)
    if logger is not None:
        logger.info('Calculating optimal thresholds for quantization using KL divergence'
                    ' with num_quantized_bins=%d' % num_quantized_bins)
    th_dict = {}
    # copy hist_dict keys since the keys() only returns a view in python3
    layer_names = list(hist_dict.keys())
    for name in layer_names:
        assert name in hist_dict
        min_val, max_val, th, divergence = \
            _get_optimal_threshold(hist_dict[name], quantized_dtype,
                                   num_quantized_bins=num_quantized_bins)
        if min_val >= 0 and quantized_dtype in ['auto', 'uint8']:
            th_dict[name] = (0, th)
        else:
            th_dict[name] = (-th, th)
        del hist_dict[name]  # release the memory
        if logger:
            logger.debug('layer=%s, min_val=%f, max_val=%f, th=%f, divergence=%f'
                         % (name, min_val, max_val, th, divergence))
    return th_dict


def _load_sym(sym, logger=None):
    """Given a str as a path the symbol .json file or a symbol, returns a Symbol object."""
    if isinstance(sym, str):  # sym is a symbol file path
        cur_path = os.path.dirname(os.path.realpath(__file__))
        symbol_file_path = os.path.join(cur_path, sym)
        if logger:
            logger.info('Loading symbol from file %s' % symbol_file_path)
        return sym_load(symbol_file_path)
    elif isinstance(sym, Symbol):
        return sym
    else:
        raise ValueError('_load_sym only accepts Symbol or path to the symbol file,'
                         ' while received type %s' % str(type(sym)))


def _load_params(params, logger=None):
    """Given a str as a path to the .params file or a pair of params,
    returns two dictionaries representing arg_params and aux_params.
    """
    if isinstance(params, str):
        cur_path = os.path.dirname(os.path.realpath(__file__))
        param_file_path = os.path.join(cur_path, params)
        if logger:
            logger.info('Loading params from file %s' % param_file_path)
        save_dict = nd_load(param_file_path)
        arg_params = {}
        aux_params = {}
        for k, v in save_dict.items():
            tp, name = k.split(':', 1)
            if tp == 'arg':
                arg_params[name] = v
            if tp == 'aux':
                aux_params[name] = v
        return arg_params, aux_params
    elif isinstance(params, (tuple, list)) and len(params) == 2:
        return params[0], params[1]
    else:
        raise ValueError('Unsupported params provided. Must be either a path to the param file or'
                         ' a pair of dictionaries representing arg_params and aux_params')

# pylint: disable=super-init-not-called
class _DataIterWrapper(DataIter):
    """DataIter wrapper for general iterator, e.g., gluon dataloader"""
    def __init__(self, calib_data):
        self._data = calib_data
        try:
            calib_iter = iter(calib_data)
        except TypeError as e:
            raise TypeError('calib_data is not a valid iterator. {}'.format(str(e)))
        data_example = next(calib_iter)
        if isinstance(data_example, (list, tuple)):
            data_example = list(data_example)
        else:
            data_example = [data_example]
        # suppose there must be one label in data_example
        # TODO(xinyu-intel): little tricky here, need to refactor.
        num_data = len(data_example)
        assert num_data > 0
        # here reshape is to handle the 5D/6D input data
        if len(data_example[0].shape) > 4:
            data_example[0] = data_example[0].reshape((-1,) + data_example[0].shape[2:])
        self.provide_data = [DataDesc(name='data', shape=(data_example[0].shape))]
        self.provide_data += [DataDesc(name='data{}'.format(i), shape=x.shape) for i, x in enumerate(data_example[1:])]
        # data0, data1, ..., label
        if num_data >= 3:
            self.provide_data = [DataDesc(name='data{}'.format(i), shape=x.shape)
                                 for i, x in enumerate(data_example[0:])]
        self.batch_size = data_example[0].shape[0]
        self.reset()

    def reset(self):
        self._iter = iter(self._data)

    def next(self):
        next_data = next(self._iter)
        # here reshape is to handle the 5D/6D input data
        if len(next_data[0].shape) > 4:
            next_data[0] = next_data[0].reshape((-1,) + next_data[0].shape[2:])
        return DataBatch(data=next_data)
# pylint: enable=super-init-not-called

def _as_data_iter(calib_data):
    """Convert normal iterator to mx.io.DataIter while parsing the data_shapes"""
    if isinstance(calib_data, DataIter):
        # already validated DataIter, just return
        return calib_data, calib_data.provide_data

    calib_data = _DataIterWrapper(calib_data)
    return calib_data, calib_data.provide_data

def quantize_model(sym, arg_params, aux_params,
                   data_names=('data',), label_names=('softmax_label',),
                   ctx=cpu(), excluded_sym_names=None, excluded_op_names=None, calib_mode='entropy',
                   calib_data=None, num_calib_examples=None,
                   quantized_dtype='int8', quantize_mode='smart',
                   quantize_granularity='tensor-wise', logger=None):
    """User-level API for generating a quantized model from a FP32 model w/ or w/o calibration.
    The backend quantized operators are only enabled for Linux systems. Please do not run
    inference using the quantized models on Windows for now.
    The quantization implementation adopts the TensorFlow's approach:
    https://www.tensorflow.org/performance/quantization.
    The calibration implementation borrows the idea of Nvidia's 8-bit Inference with TensorRT:
    http://on-demand.gputechconf.com/gtc/2017/presentation/s7310-8-bit-inference-with-tensorrt.pdf
    and adapts the method to MXNet.

    Parameters
    ----------
    sym : str or Symbol
        Defines the structure of a neural network for FP32 data types.
    arg_params : dict
        Dictionary of name to `NDArray`.
    aux_params : dict
        Dictionary of name to `NDArray`.
    data_names : a list of strs
        Data names required for creating a Module object to run forward propagation on the
        calibration dataset.
    label_names : a list of strs
        Label names required for creating a Module object to run forward propagation on the
        calibration dataset.
    ctx : Context
        Defines the device that users want to run forward propagation on the calibration
        dataset for collecting layer output statistics. Currently, only supports single context.
    excluded_sym_names : list of strings
        A list of strings representing the names of the symbols that users want to excluding
        from being quantized.
    excluded_op_names : list of strings
        A list of strings representing the names of the operators that users want to excluding
        from being quantized.
    calib_mode : str
        If calib_mode='none', no calibration will be used and the thresholds for
        requantization after the corresponding layers will be calculated at runtime by
        calling min and max operators. The quantized models generated in this
        mode are normally 10-20% slower than those with calibrations during inference.
        If calib_mode='naive', the min and max values of the layer outputs from a calibration
        dataset will be directly taken as the thresholds for quantization.
        If calib_mode='entropy' (default mode), the thresholds for quantization will be
        derived such that the KL divergence between the distributions of FP32 layer outputs and
        quantized layer outputs is minimized based upon the calibration dataset.
    calib_data : DataIter
        A data iterator initialized by the calibration dataset.
    num_calib_examples : int or None
        The maximum number of examples that user would like to use for calibration. If not provided,
        the whole calibration dataset will be used.
    quantized_dtype : str
        The quantized destination type for input data. Currently support 'int8', 'uint8' and 'auto'.
        'auto' means automatically select output type according to calibration result.
        Default value is 'int8'.
    quantize_mode : str
        The mode that quantization pass to apply. Support 'full' and 'smart'.
        'full' means quantize all operator if possible.
        'smart' means quantization pass will smartly choice which operator should be quantized.
    quantize_granularity: str
        The granularity of quantization, currently supports 'tensor-wise' and 'channel-wise'
        quantization. The default value is 'tensor-wise'.
    logger : Object
        A logging object for printing information during the process of quantization.

    Returns
    -------
    tuple
        A tuple of quantized symbol, quantized arg_params, and aux_params.
    -------
    """
    if excluded_sym_names is None:
        excluded_sym_names = []
    if not isinstance(excluded_sym_names, list):
        raise ValueError('excluded_sym_names must be a list of strings representing'
                         ' the names of the symbols that will not be quantized,'
                         ' while received type %s' % str(type(excluded_sym_names)))

    if excluded_op_names is None:
        excluded_op_names = []
    if not isinstance(excluded_op_names, list):
        raise ValueError('excluded_op_names must be a list of strings representing'
                         ' the names of the operators that will not be quantized,'
                         ' while received type %s' % str(type(excluded_op_names)))

    if logger:
        os.environ['MXNET_QUANTIZATION_VERBOSE'] = '1'
        logger.info('Quantizing symbol')
    if quantized_dtype not in ('int8', 'uint8', 'auto'):
        raise ValueError('unknown quantized_dtype %s received,'
                         ' expected `int8`, `uint8` or `auto`' % quantized_dtype)
    if quantize_granularity not in ('tensor-wise', 'channel-wise'):
        raise ValueError('unkonwn quantize_granularity %s received,'
                         ' expected `tensor-wise` or `channel-wise`.' % quantize_granularity)
    qsym, calib_layer = _quantize_symbol(sym, ctx, excluded_symbols=excluded_sym_names,
                                         excluded_operators=excluded_op_names,
                                         offline_params=list(arg_params.keys()),
                                         quantized_dtype=quantized_dtype,
                                         quantize_mode=quantize_mode,
                                         quantize_granularity=quantize_granularity)
    th_dict = {}
    if calib_mode is not None and calib_mode != 'none':
        if not isinstance(ctx, Context):
            raise ValueError('currently only supports single ctx, while received %s' % str(ctx))
        if calib_data is None:
            raise ValueError('calib_data must be provided when calib_mode=%s' % calib_mode)
        if not isinstance(calib_data, DataIter):
            raise ValueError('calib_data must be of DataIter type when calib_mode=%s,'
                             ' while received type %s' % (calib_mode, str(type(calib_data))))

        mod = Module(symbol=sym, data_names=data_names, label_names=label_names, context=ctx)
        if len(calib_data.provide_label) > 0:
            mod.bind(for_training=False, data_shapes=calib_data.provide_data,
                     label_shapes=calib_data.provide_label)
        else:
            mod.bind(for_training=False, data_shapes=calib_data.provide_data)
        mod.set_params(arg_params, aux_params)
        if calib_mode == 'entropy':
            hist_dict, num_examples = _collect_layer_histogram(mod, calib_data,
                                                               include_layer=calib_layer,
                                                               max_num_examples=num_calib_examples,
                                                               logger=logger)
            if logger:
                logger.info('Collected layer outputs from FP32 model using %d examples' % num_examples)
                logger.info('Calculating optimal thresholds for quantization')
            th_dict = _get_optimal_thresholds(hist_dict, quantized_dtype, logger=logger)
        elif calib_mode == 'naive':
            th_dict, num_examples = _collect_layer_output_min_max(
                mod, calib_data, quantized_dtype, include_layer=calib_layer, max_num_examples=num_calib_examples,
                logger=logger)
            if logger:
                logger.info('Collected layer output min/max values from FP32 model using %d examples'
                            % num_examples)
        else:
            raise ValueError('unknown calibration mode %s received,'
                             ' expected `none`, `naive`, or `entropy`' % calib_mode)
        qsym = _calibrate_quantized_sym(qsym, th_dict)

    if logger:
        logger.info('Quantizing parameters')
    qarg_params = _quantize_params(qsym, arg_params, th_dict)

    return qsym, qarg_params, aux_params

def quantize_model_mkldnn(sym, arg_params, aux_params,
                          data_names=('data',), label_names=('softmax_label',),
                          ctx=cpu(), excluded_sym_names=None, excluded_op_names=None,
                          calib_mode='entropy', calib_data=None, num_calib_examples=None,
                          quantized_dtype='int8', quantize_mode='smart',
                          quantize_granularity='tensor-wise', logger=None):
    """User-level API for generating a fusion + quantized model from a FP32 model
    w/ or w/o calibration with Intel MKL-DNN.
    The backend quantized operators are only enabled for Linux systems. Please do not run
    inference using the quantized models on Windows for now.

    Parameters
    ----------
    same with quantize_model

    Returns
    -------
    tuple
        A tuple of quantized symbol, quantized arg_params, and aux_params.
    -------
    """
    if not isinstance(ctx, Context):
        raise ValueError('currently only supports single ctx, while received %s' % str(ctx))
    if ctx.device_type != 'cpu':
        raise ValueError(
            'quantize_model_mkldnn only support Intel cpu platform with MKL-DNN Backend')

    sym = sym.get_backend_symbol('MKLDNN_QUANTIZE')

    qsym, qarg_params, aux_params = quantize_model(sym=sym, arg_params=arg_params, aux_params=aux_params,
                                                   data_names=data_names, label_names=label_names,
                                                   ctx=ctx, excluded_sym_names=excluded_sym_names,
                                                   excluded_op_names=excluded_op_names,
                                                   calib_mode=calib_mode, calib_data=calib_data,
                                                   num_calib_examples=num_calib_examples,
                                                   quantized_dtype=quantized_dtype, quantize_mode=quantize_mode,
                                                   quantize_granularity=quantize_granularity, logger=logger)

    qsym = qsym.get_backend_symbol('MKLDNN_QUANTIZE')

    return qsym, qarg_params, aux_params

def quantize_graph(sym, arg_params, aux_params, ctx=cpu(),
                   excluded_sym_names=None, excluded_op_names=None,
                   calib_mode='entropy', quantized_dtype='int8',
                   quantize_mode='full', quantize_granularity='tensor-wise',
                   LayerOutputCollector=None, logger=None):
    """User-level API for generating a quantized model from a FP32 model w/o calibration
    and a collector for naive or entropy calibration.
    The backend quantized operators are only enabled for Linux systems. Please do not run
    inference using the quantized models on Windows for now.
    Parameters
    ----------
    sym : str or Symbol
        Defines the structure of a neural network for FP32 data types.
    ctx : Context
        Defines the device that users want to run forward propagation on the calibration
        dataset for collecting layer output statistics. Currently, only supports single context.
    arg_params : dict
        Dictionary of name to `NDArray`.
    aux_params : dict
        Dictionary of name to `NDArray`.
    excluded_sym_names : list of strings
        A list of strings representing the names of the symbols that users want to excluding
        from being quantized.
    excluded_op_names : list of strings
        A list of strings representing the names of the operators that users want to excluding
    calib_mode : str
        If calib_mode='none', no calibration will be used and the thresholds for
        requantization after the corresponding layers will be calculated at runtime by
        calling min and max operators. The quantized models generated in this
        mode are normally 10-20% slower than those with calibrations during inference.
        If calib_mode='naive', the min and max values of the layer outputs from a calibration
        dataset will be directly taken as the thresholds for quantization.
        If calib_mode='entropy' (default mode), the thresholds for quantization will be
        derived such that the KL divergence between the distributions of FP32 layer outputs and
        quantized layer outputs is minimized based upon the calibration dataset.
    quantized_dtype : str
        The quantized destination type for input data. Currently support 'int8'
        , 'uint8' and 'auto'. 'auto' means automatically select output type according to calibration result.
        Default value is 'int8'.
    quantize_mode : str
        The mode that quantization pass to apply. Support 'full' and 'smart'.
        'full' means quantize all operator if possible.
        'smart' means quantization pass will smartly choice which operator should be quantized.
    quantize_granularity: str
        The granularity of quantization, currently supports 'tensor-wise' and 'channel-wise'
        quantization. The default value is 'tensor-wise'.
    LayerOutputCollector : class
        For customize calibration method usage.
    logger : Object
        A logging object for printing information during the process of quantization.
    Returns
    -------
    tuple
        A tuple of quantized symbol, quantized arg_params, aux_params and collector.
    -------
    """
    if excluded_sym_names is None:
        excluded_sym_names = []
    if not isinstance(excluded_sym_names, list):
        raise ValueError('excluded_sym_names must be a list of strings representing'
                         ' the names of the symbols that will not be quantized,'
                         ' while received type %s' % str(type(excluded_sym_names)))
    if not isinstance(ctx, Context):
        raise ValueError('currently only supports single ctx, while received %s' % str(ctx))
    if logger:
        os.environ['MXNET_QUANTIZATION_VERBOSE'] = '1'
        logger.info('Quantizing graph')
    if quantized_dtype not in ('int8', 'uint8', 'auto'):
        raise ValueError('unknown quantized_dtype %s received,'
                         ' expected `int8`, `uint8` or `auto`' % quantized_dtype)
    if quantize_granularity not in ('tensor-wise', 'channel-wise'):
        raise ValueError('unkonwn quantize_granularity %s received,'
                         ' expected `tensor-wise` or `channel-wise`.' % quantize_granularity)
    qsym, calib_layer = _quantize_symbol(sym, ctx, excluded_symbols=excluded_sym_names,
                                         excluded_operators=excluded_op_names,
                                         offline_params=list(
                                             arg_params.keys()),
                                         quantized_dtype=quantized_dtype,
                                         quantize_mode=quantize_mode,
                                         quantize_granularity=quantize_granularity)

    th_dict = {}
    collector = None
    if calib_mode is not None and calib_mode != 'none':
        if calib_mode == 'entropy':
            collector = _LayerHistogramCollector(
                include_layer=calib_layer, logger=logger)
            if logger:
                logger.info(
                    'Create a layer output collector for entropy calibration.')
        elif calib_mode == 'naive':
            collector = _LayerOutputMinMaxCollector(quantized_dtype=quantized_dtype,
                                                    include_layer=calib_layer, logger=logger)
            if logger:
                logger.info(
                    'Create a layer output minmax collector for naive calibration')
        elif calib_mode == 'customize' and LayerOutputCollector is not None:
            collector = LayerOutputCollector
            if logger:
                logger.info(
                    'Create a customize layer output minmax collector for calibration')
        else:
            raise ValueError('unknown calibration mode %s received,'
                             ' expected `none`, `naive`, `entropy` or `customize`' % calib_mode)
        if logger:
            logger.info('Collector created, please use set_monitor_callback'
                        ' to collect calibration information.')

    if logger:
        logger.info('Quantizing parameters')
    qarg_params = _quantize_params(qsym, arg_params, th_dict)

    return qsym, qarg_params, aux_params, collector

def calib_graph(qsym, arg_params, aux_params, collector,
                calib_mode='entropy', quantized_dtype='int8', logger=logging):
    """User-level API for calibrating a quantized model using a filled collector.
    The backend quantized operators are only enabled for Linux systems. Please do not run
    inference using the quantized models on Windows for now.
    Parameters
    ----------
    qsym : str or Symbol
        Defines the structure of a neural network for INT8 data types.
    arg_params : dict
        Dictionary of name to `NDArray`.
    aux_params : dict
        Dictionary of name to `NDArray`.
    collector : function
        layer collector for naive or entropy calibration.
    calib_mode : str
        If calib_mode='none', no calibration will be used and the thresholds for
        requantization after the corresponding layers will be calculated at runtime by
        calling min and max operators. The quantized models generated in this
        mode are normally 10-20% slower than those with calibrations during inference.
        If calib_mode='naive', the min and max values of the layer outputs from a calibration
        dataset will be directly taken as the thresholds for quantization.
        If calib_mode='entropy' (default mode), the thresholds for quantization will be
        derived such that the KL divergence between the distributions of FP32 layer outputs and
        quantized layer outputs is minimized based upon the calibration dataset.
    quantized_dtype : str
        The quantized destination type for input data. Currently support 'int8'
        , 'uint8' and 'auto'. 'auto' means automatically select output type according to calibration result.
        Default value is 'int8'.
    logger : Object
        A logging object for printing information during the process of quantization.
    Returns
    -------
    tuple
        A tuple of calibrated symbol, quantized arg_params, aux_params.
    -------
    """
    th_dict = {}
    if calib_mode is not None and calib_mode != 'none':
        if calib_mode == 'entropy':
            if logger:
                logger.info('Calculating optimal thresholds for quantization')
            th_dict = _get_optimal_thresholds(
                collector.hist_dict, quantized_dtype, logger=logger)
        elif calib_mode == 'naive':
            th_dict = collector.min_max_dict
        elif calib_mode == 'customize':
            th_dict = collector.min_max_dict
        else:
            raise ValueError('unknown calibration mode %s received,'
                             ' expected `none`, `naive`, `entropy` or `customize`' % calib_mode)
        qsym = _calibrate_quantized_sym(qsym, th_dict)
    else:
        raise ValueError('please set calibration mode to naive or entropy.')

    if logger:
        logger.info('Quantizing parameters')
    qarg_params = _quantize_params(qsym, arg_params, th_dict)

    return qsym, qarg_params, aux_params

def quantize_net_v2(network, quantized_dtype='auto', quantize_mode='full', quantize_granularity='tensor-wise',
                    exclude_layers=None, exclude_layers_match=None, exclude_operators=None,
                    calib_data=None, data_shapes=None, calib_mode='none',
                    num_calib_examples=None, ctx=cpu(), LayerOutputCollector=None, logger=None):
    """User-level API for Gluon users to generate a quantized SymbolBlock from a FP32 HybridBlock w/ or w/o calibration.
    The backend quantized operators are only enabled for Linux systems. Please do not run
    inference using the quantized models on Windows for now.

    Parameters
    ----------
    network : Gluon HybridBlock
        Defines the structure of a neural network for FP32 data types.
    quantized_dtype : str
        The quantized destination type for input data. Currently support 'int8'
        , 'uint8' and 'auto'. 'auto' means automatically select output type according to calibration result.
        Default value is 'int8'.
    quantize_mode : str
        The mode that quantization pass to apply. Support 'full' and 'smart'.
        'full' means quantize all operator if possible.
        'smart' means quantization pass will smartly choice which operator should be quantized.
    quantize_granularity: str
        The granularity of quantization, currently supports 'tensor-wise' and 'channel-wise'
        quantization. The default value is 'tensor-wise'.
    exclude_layers : list of strings
        A list of strings representing the names of the symbols that users want to excluding
    exclude_layers_match : list of strings
        A list of strings wildcard matching the names of the symbols that users want to excluding
        from being quantized.
    exclude_operators : list of strings
        A list of strings representing the names of the operators that users want to excluding
    calib_data : mx.io.DataIter or gluon.DataLoader
        A iterable data loading object.
    data_shapes : list
        List of DataDesc, required if calib_data is not provided
    calib_mode : str
        If calib_mode='none', no calibration will be used and the thresholds for
        requantization after the corresponding layers will be calculated at runtime by
        calling min and max operators. The quantized models generated in this
        mode are normally 10-20% slower than those with calibrations during inference.
        If calib_mode='naive', the min and max values of the layer outputs from a calibration
        dataset will be directly taken as the thresholds for quantization.
        If calib_mode='entropy' (default mode), the thresholds for quantization will be
        derived such that the KL divergence between the distributions of FP32 layer outputs and
        quantized layer outputs is minimized based upon the calibration dataset.
    num_calib_examples : int or None
        The maximum number of examples that user would like to use for calibration. If not provided,
        the whole calibration dataset will be used.
    ctx : Context
        Defines the device that users want to run forward propagation on the calibration
        dataset for collecting layer output statistics. Currently, only supports single context.
    LayerOutputCollector : class
        For customize calibration method usage.
    logger : Object
        A logging object for printing information during the process of quantization.

    Returns
    -------
    network : Gluon SymbolBlock
        Defines the structure of a neural network for INT8 data types.
    -------
    """

    if logger:
        logger.info('Export HybridBlock')
    network.hybridize()
    import mxnet as mx
    if calib_data is not None:
        if isinstance(calib_data, DataIter):
            dshapes = calib_data.provide_data
        else:
            calib_data, dshapes = _as_data_iter(calib_data)
        if not data_shapes:
            data_shapes = dshapes
    if not data_shapes:
        raise ValueError('data_shapes required')
    data_nd = []
    for shape in data_shapes:
        data_nd.append(mx.nd.zeros(shape.shape))
    while True:
        try:
            network(*data_nd)
        except TypeError:
            del data_nd[-1]
            del calib_data.provide_data[-1]
            continue
        else:
            break

    import tempfile
    try:
        from tempfile import TemporaryDirectory
    except ImportError:
        # really simple implementation of TemporaryDirectory
        class TemporaryDirectory(object):
            def __init__(self, suffix='', prefix='', dir=''):
                self._dirname = tempfile.mkdtemp(suffix, prefix, dir)

            def __enter__(self):
                return self._dirname

            def __exit__(self, exc_type, exc_value, traceback):
                shutil.rmtree(self._dirname)
    # TODO(xinyu-intel): tmp solution to save and reload for mxnet.mod.Module.
    # will enhance `export` function to return `sym, args, auxs` directly.
    with TemporaryDirectory() as tmpdirname:
        prefix = os.path.join(tmpdirname, 'tmp')
        network.export(prefix, epoch=0)
        symnet, args, auxs = mx.model.load_checkpoint(prefix, 0)

    if exclude_layers is None:
        exclude_layers = []
    if exclude_layers_match is None:
        exclude_layers_match = []
    if exclude_operators is None:
        exclude_operators = []
    for name_match in exclude_layers_match:
        for layers in list(symnet.get_internals()):
            if layers.name.find(name_match) != -1:
                exclude_layers.append(layers.name)
    if logger:
        logger.info('These layers have been excluded %s' % exclude_layers)

    if ctx == mx.cpu():
        symnet = symnet.get_backend_symbol('MKLDNN_QUANTIZE')

    qsym, qarg_params, aux_params, collector = quantize_graph(
        sym=symnet, arg_params=args, aux_params=auxs, ctx=ctx,
        excluded_sym_names=exclude_layers, excluded_op_names=exclude_operators,
        calib_mode=calib_mode, quantized_dtype=quantized_dtype, quantize_mode=quantize_mode,
        quantize_granularity=quantize_granularity, LayerOutputCollector=LayerOutputCollector,
        logger=logger)

    if calib_mode is not None and calib_mode != 'none':
        if not isinstance(ctx, Context):
            raise ValueError(
                'currently only supports single ctx, while received %s' % str(ctx))
        if calib_data is None:
            raise ValueError(
                'calib_data must be provided when calib_mode=%s' % calib_mode)
        if calib_mode in ['naive', 'entropy', 'customize']:
            data_names = [pair[0] for pair in calib_data.provide_data]
            mod = Module(symbol=symnet, context=ctx,
                         data_names=data_names, label_names=None)
            mod.bind(for_training=False, data_shapes=data_shapes)
            mod.set_params(args, auxs, allow_missing=False, force_init=True)
            num_examples = _collect_layer_statistics(mod, calib_data, collector,
                                                     num_calib_examples, logger)
            if logger:
                logger.info('Collected layer output values from FP32 model using %d examples'
                            % num_examples)
            qsym, qarg_params, aux_params = calib_graph(
                qsym=qsym, arg_params=args, aux_params=auxs, collector=collector,
                calib_mode=calib_mode, quantized_dtype=quantized_dtype, logger=logger)
        else:
            raise ValueError(
                'please set calibration mode to naive or entropy.')
    elif calib_mode is not None and calib_mode == 'none':
        data_names = [pair[0] for pair in data_shapes]

    if ctx == mx.cpu():
        qsym = qsym.get_backend_symbol('MKLDNN_QUANTIZE')

    from ..gluon import SymbolBlock
    data_sym = []
    for name in data_names:
        data_sym.append(mx.sym.var(name))
    net = SymbolBlock(qsym, data_sym)
    # TODO(xinyu-intel): tmp solution to save param_dict and reload for SymbolBlock
    # will enhance SymbolBlock to load args, auxs directly.
    with TemporaryDirectory() as tmpdirname:
        prefix = os.path.join(tmpdirname, 'tmp')
        param_name = '%s-%04d.params' % (prefix + 'net-quantized', 0)
        save_dict = {('arg:%s' % k): v.as_in_context(cpu())
                     for k, v in qarg_params.items()}
        save_dict.update({('aux:%s' % k): v.as_in_context(cpu())
                          for k, v in aux_params.items()})
        nd_save(param_name, save_dict)
        net.collect_params().load(param_name, cast_dtype=True, dtype_source='saved')
        net.collect_params().reset_ctx(ctx)
    return net

def quantize_net(network, quantized_dtype='auto', quantize_mode='full',
                 exclude_layers=None, exclude_layers_match=None, exclude_operators=None,
                 calib_data=None, data_shapes=None, calib_mode='none',
                 num_calib_examples=None, ctx=cpu(), logger=None):
    """User-level API for Gluon users to generate a quantized SymbolBlock from a FP32 HybridBlock w/ or w/o calibration.
       Will be deprecated after MXNet 2.0, please use quantize_net_v2.
    """
    warnings.warn('WARNING: This will be deprecated after MXNet 2.0, please use quantize_net_v2.')
    return quantize_net_v2(network=network, quantized_dtype=quantized_dtype,
                           quantize_mode=quantize_mode,
                           quantize_granularity='tensor-wise',
                           exclude_layers=exclude_layers,
                           exclude_layers_match=exclude_layers_match,
                           exclude_operators=exclude_operators,
                           calib_data=calib_data, data_shapes=data_shapes,
                           calib_mode=calib_mode, num_calib_examples=num_calib_examples,
                           ctx=ctx, LayerOutputCollector=None, logger=logger)
