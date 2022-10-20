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

import abc
import ctypes
import logging
import os
import warnings
import numpy as np
import mxnet as mx
from ..base import _LIB, check_call, py_str
from ..base import c_array, c_str, mx_uint, mx_real_t, c_str_array
from ..base import SymbolHandle
from ..symbol import Symbol
from .. import ndarray
from ..io import DataDesc
from ..device import cpu, Device
from ..util import is_np_array, wrap_ctx_to_device_func


def _multilist_iterator(arg, func):
    """Iterate over multidiemnsional list and returns new list
    with same dimensions, but applied `func` function on list elements.
    E.g. _multilist_iterator([1, 2, [3, 4]], lambda x: x**2) = [1, 4, [9, 16]]
    """
    ret = []
    if isinstance(arg, list):
        for el in arg:
            ret.append(_multilist_iterator(el, func))
    else:
        return func(arg)

    return ret

def _quantize_params(qsym, params, min_max_dict):
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
    min_max_dict : dict of min/max pairs of layers' output
    """
    inputs_name = qsym.list_arguments()
    quantized_params = {}
    if is_np_array():
        quantize_fn = mx.npx.contrib_quantize
        min_fn = lambda arr: mx.np.array([mx.np.min(arr)])
        max_fn = lambda arr: mx.np.array([mx.np.max(arr)])
        array_cls = mx.np
    else:
        quantize_fn = mx.nd.contrib.quantize
        min_fn = mx.nd.min
        max_fn = mx.nd.max
        array_cls = mx.nd

    for name in inputs_name:
        if name.endswith(('weight_quantize', 'bias_quantize')):
            original_name = name[:-len('_quantize')]
            param = params[original_name]
            # pylint: disable=unbalanced-tuple-unpacking
            param_min = min_fn(param)
            param_max = max_fn(param)
            val, vmin, vmax = quantize_fn(data=param,
                                          min_range=param_min,
                                          max_range=param_max,
                                          out_type='int8')
            quantized_params[name] = val
            quantized_params[name+'_min'] = vmin
            quantized_params[name+'_max'] = vmax
        elif name in params:
            quantized_params[name] = params[name]
        elif name.endswith(('_min')):
            output = name[: - len('_min')]
            if output in min_max_dict:
                quantized_params[name] = array_cls.array([min_max_dict[output][0]])
        elif name.endswith(('_max')):
            output = name[: - len('_min')]
            if output in min_max_dict:
                quantized_params[name] = array_cls.array([min_max_dict[output][1]])
    return quantized_params


def _quantize_symbol(sym, device, excluded_symbols=None, excluded_operators=None,
                     offline_params=None, quantized_dtype='int8', quantize_mode='smart',
                     quantize_granularity='tensor-wise'):
    """Given a symbol object representing a neural network of data type FP32,
    quantize it into a INT8 network.

    Parameters
    ----------
    sym : Symbol
        FP32 neural network symbol.
    device : Device
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
    quantized_dtype : str
        The quantized destination type for input data.
    quantize_mode : str
        The mode that quantization pass to apply.
    quantize_granularity : str
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
                                     ctypes.byref(ctypes.c_int(device.device_typeid)),
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
    calib_layers = []
    calib_layers = [py_str(calib_str[i]) for i in range(size.value)]
    return Symbol(out), calib_layers


class CalibrationCollector(object):
    """Base class for all other collectors used with quantization"""
    __metaclass__ = abc.ABCMeta

    def __init__(self):
        self.include_layers = None
        self.min_max_dict = {}

    @abc.abstractmethod
    def collect(self, name, op_name, arr):
        """Function which is registered to Block as monitor callback. Names of layers
        requiring calibration are stored in `self.include_layers` variable.

        Parameters
        ----------
        name : str
            Node name from which collected data comes from.
        op_name : str
            Operator name from which collected data comes from. Single operator
            can have multiple input/ouput nodes - each should have different name.
        arr : NDArray
            NDArray containing data of monitored node.
        """

    def post_collect(self):
        """ Function called after collecting parameters. Returns dictionary of min and max values
        for each calibrated layer. If not overriden, returns content of `self.min_max_dict`.
        """
        return self.min_max_dict


class _LayerHistogramCollector(CalibrationCollector):
    """Saves layer histogram in a dict with layer names as keys and lists of NDArrays as
    values. The collected histogram will be used for calculating the optimal thresholds for
    quantization using KL divergence.
    """
    def __init__(self, quantized_dtype, num_bins=8001, include_layers=None, logger=None):
        super(_LayerHistogramCollector, self).__init__()
        self.hist_dict = {}
        self.num_bins = num_bins
        self.include_layers = include_layers
        self.logger = logger
        self.quantized_dtype = quantized_dtype

    def collect(self, name, op_name, arr):
        """Callback function for collecting layer output NDArrays."""
        if name not in self.include_layers:
            return
        arr = arr.copyto(cpu()).asnumpy()
        if self.logger:
            self.logger.debug(f"Collecting layer {name} histogram of shape {arr.shape}")
        min_range = np.min(arr)
        max_range = np.max(arr)
        th = max(abs(min_range), abs(max_range))
        if name in self.hist_dict:
            self.hist_dict[name] = self.combine_histogram(self.hist_dict[name], arr, min_range, max_range, th)
        else:
            hist, hist_edges = np.histogram(arr, bins=self.num_bins, range=(-th, th))
            self.hist_dict[name] = (hist, hist_edges, min_range, max_range, th)

    def post_collect(self):
        min_max_dict = self.get_optimal_thresholds(self.hist_dict, self.quantized_dtype, logger=self.logger)
        return min_max_dict

    @staticmethod
    def combine_histogram(old_hist, arr, new_min, new_max, new_th):
        """Collect layer histogram for arr and combine it with old histogram."""
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

    # pylint: disable=line-too-long
    @staticmethod
    def get_optimal_threshold(hist_data, quantized_dtype, num_quantized_bins=255):
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

    @staticmethod
    def get_optimal_thresholds(hist_dict, quantized_dtype, num_quantized_bins=255, logger=None):
        """Given a ndarray dict, find the optimal threshold for quantizing each value of the key."""
        assert isinstance(hist_dict, dict)
        if logger is not None:
            logger.info('Calculating optimal thresholds for quantization using KL divergence'
                        f' with num_quantized_bins={num_quantized_bins}')
        th_dict = {}
        # copy hist_dict keys since the keys() only returns a view in python3
        layer_names = list(hist_dict.keys())
        for name in layer_names:
            assert name in hist_dict
            min_val, max_val, th, divergence = \
                _LayerHistogramCollector.get_optimal_threshold(hist_dict[name], quantized_dtype,
                                                               num_quantized_bins=num_quantized_bins)
            if min_val >= 0 and quantized_dtype in ['auto', 'uint8']:
                th_dict[name] = (0, th)
            else:
                th_dict[name] = (-th, th)
            del hist_dict[name]  # release the memory
            if logger:
                logger.debug(f"layer={name}, min_val={min_val}, max_val={max_val}, th={th}, divergence={divergence}")
        return th_dict


class _LayerOutputMinMaxCollector(CalibrationCollector):
    """Saves layer output min and max values in a dict with layer names as keys.
    The collected min and max values will be directly used as thresholds for quantization.
    """
    def __init__(self, quantized_dtype, include_layers=None, logger=None):
        super(_LayerOutputMinMaxCollector, self).__init__()
        self.min_max_dict = {}
        self.quantized_dtype = quantized_dtype
        self.include_layers = include_layers
        self.logger = logger

    def collect(self, name, op_name, arr):
        """Callback function for collecting min and max values from an NDArray."""
        if name not in self.include_layers:
            return
        arr = arr.copyto(cpu()).asnumpy()
        min_range = np.min(arr)
        max_range = np.max(arr)
        if name in self.min_max_dict:
            cur_min_max = self.min_max_dict[name]
            self.min_max_dict[name] = (min(cur_min_max[0], min_range),
                                       max(cur_min_max[1], max_range))
        else:
            self.min_max_dict[name] = (min_range, max_range)
        if self.logger:
            self.logger.debug(f"Collecting layer {name} min_range={min_range}, max_range={max_range}")


def _calibrate_quantized_sym(qsym, min_max_dict):
    """Given a dictionary containing the thresholds for quantizing the layers,
    set the thresholds into the quantized symbol as the params of requantize operators.
    """
    if min_max_dict is None or len(min_max_dict) == 0:
        return qsym
    num_layer_outputs = len(min_max_dict)
    layer_output_names = []
    min_vals = []
    max_vals = []
    for k, v in min_max_dict.items():
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


def _collect_layer_statistics(sym_block, data, collector, num_inputs, num_calib_batches=None, logger=None):
    if not isinstance(data, mx.gluon.data.DataLoader):
        raise ValueError(f'Only supports data as a type of DataLoader, while received type {str(type(data))}')
    sym_block.register_op_hook(collector.collect, monitor_all=True)
    num_batches = 0
    for batch in data:
        if not isinstance(batch, list):
            batch = [batch]
        batch = _multilist_iterator(batch, lambda b: b.as_in_context(mx.cpu()))
        sym_block(*batch[:num_inputs])
        num_batches += 1
        if num_calib_batches is not None and num_batches >= num_calib_batches:
            break
    if logger is not None:
        logger.info(f"Collected statistics from {num_batches} batches")
    return num_batches


def _generate_list_of_data_desc(data_shapes, data_types):
    """Convert list of tuples to list of DataDesc."""
    def flatten_list(arg):
        ret = []
        for el in arg:
            if isinstance(el, list):
                ret += flatten_list(el)
            else:
                ret.append(el)
        return ret

    flattened_data_types = flatten_list(data_types)
    flattened_data_shapes = flatten_list(data_shapes)

    if all(isinstance(x, DataDesc) for x in flattened_data_shapes):
        return data_shapes

    assert len(flattened_data_types) == len(flattened_data_shapes)

    # pass integral type as reference
    counter = [0]
    def get_data_desc(data_shape, counter=counter, data_types=flattened_data_types):
        if isinstance(data_shape, DataDesc):
            return data_shape
        elif isinstance(data_shape, tuple):
            desc = DataDesc(name='data' + str(counter[0]), shape=data_shape,
                                        dtype=data_types[counter[0]])
            counter[0] += 1
            return desc
        else:
            raise ValueError('data_shapes must be either a list of DataDesc or a list of Tuple')


    if len(data_shapes) == 1 and not isinstance(data_shapes[0], list):
        data_descs = [DataDesc(name='data', shape=data_shapes[0], dtype=data_types[0])]
    else:
        data_descs = _multilist_iterator(data_shapes, get_data_desc)

    return data_descs

@wrap_ctx_to_device_func
def quantize_model(sym, arg_params, aux_params, data_names=('data',),
                   device=cpu(), excluded_sym_names=None, excluded_op_names=None, calib_mode='entropy',
                   calib_data=None, num_calib_batches=None,
                   quantized_dtype='int8', quantize_mode='smart',
                   quantize_granularity='tensor-wise', logger=None):
    """User-level API for generating a quantized model from a FP32 model w/ or w/o calibration.
    The backend quantized operators are only enabled for Linux systems. Please do not run
    inference using the quantized models on Windows for now.
    The quantization implementation adopts the TensorFlow's approach:
    https://www.tensorflow.org/lite/performance/post_training_quantization.
    The calibration implementation borrows the idea of Nvidia's 8-bit Inference with TensorRT:
    http://on-demand.gputechconf.com/gtc/2017/presentation/s7310-8-bit-inference-with-tensorrt.pdf
    and adapts the method to MXNet.

    .. _`quantize_model_params`:

    Parameters
    ----------
    sym : Symbol
        Defines the structure of a neural network for FP32 data types.
    arg_params : dict
        Dictionary of name to `NDArray`.
    aux_params : dict
        Dictionary of name to `NDArray`.
    data_names : list of strings
        Data names required for creating a Module object to run forward propagation on the
        calibration dataset.
    device : Device
        Defines the device that users want to run forward propagation on the calibration
        dataset for collecting layer output statistics. Currently, only supports single device.
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
    calib_data : DataLoader
        A DataLoader initialized by the calibration dataset.
    num_calib_batches : int or None
        The maximum number of batches that user would like to use for calibration. If not provided,
        the whole calibration dataset will be used.
    quantized_dtype : str
        The quantized destination type for input data. Currently support 'int8', 'uint8' and 'auto'.
        'auto' means automatically select output type according to calibration result.
        Default value is 'int8'.
    quantize_mode : str
        The mode that quantization pass to apply. Support 'full' and 'smart'.
        'full' means quantize all operator if possible.
        'smart' means quantization pass will smartly choice which operator should be quantized.
    quantize_granularity : str
        The granularity of quantization, currently supports 'tensor-wise' and 'channel-wise'
        quantization. The default value is 'tensor-wise'.
    logger : Object
        A logging object for printing information during the process of quantization.

    Returns
    -------
    quantized_model : tuple
        A tuple of quantized symbol, quantized arg_params, and aux_params.
    """
    warnings.warn('WARNING: This will be deprecated please use quantize_net with Gluon models')
    if excluded_sym_names is None:
        excluded_sym_names = []
    if not isinstance(excluded_sym_names, list):
        raise ValueError('excluded_sym_names must be a list of strings representing'
                         ' the names of the symbols that will not be quantized,'
                         f' while received type {str(type(excluded_sym_names))}')

    if excluded_op_names is None:
        excluded_op_names = []
    if not isinstance(excluded_op_names, list):
        raise ValueError('excluded_op_names must be a list of strings representing'
                         ' the names of the operators that will not be quantized,'
                         f' while received type {str(type(excluded_op_names))}')

    if logger:
        os.environ['MXNET_QUANTIZATION_VERBOSE'] = '1'
        logger.info('Quantizing symbol')
    if quantized_dtype not in ('int8', 'uint8', 'auto'):
        raise ValueError(f'unknown quantized_dtype {quantized_dtype} received,'
                         ' expected `int8`, `uint8` or `auto`')
    if quantize_granularity not in ('tensor-wise', 'channel-wise'):
        raise ValueError(f'unkonwn quantize_granularity {quantize_granularity} received,'
                         ' expected `tensor-wise` or `channel-wise`.')
    qsym, calib_layers = _quantize_symbol(sym, device, excluded_symbols=excluded_sym_names,
                                          excluded_operators=excluded_op_names,
                                          offline_params=list(arg_params.keys()),
                                          quantized_dtype=quantized_dtype,
                                          quantize_mode=quantize_mode,
                                          quantize_granularity=quantize_granularity)
    min_max_dict = {}
    if calib_mode is not None and calib_mode != 'none':
        if not isinstance(device, Device):
            raise ValueError(f'currently only supports single device, while received {str(device)}')
        if calib_data is None:
            raise ValueError(f'calib_data must be provided when calib_mode={calib_mode}')
        if not isinstance(calib_data, mx.gluon.data.DataLoader):
            raise ValueError(f'calib_data must be of DataLoader type when calib_mode={calib_mode},'
                             f' while received type {str(type(calib_data))}')

        inputs = [mx.sym.var(dname) for dname in data_names]
        param_dict = arg_params
        param_dict.update(aux_params)
        sym_block = mx.gluon.SymbolBlock(sym, inputs)
        sym_block.load_dict(param_dict)

        if calib_mode == 'entropy':
            collector = _LayerHistogramCollector(quantized_dtype=quantized_dtype,
                                                 include_layers=calib_layers,
                                                 logger=logger)
        elif calib_mode == 'naive':
            collector = _LayerOutputMinMaxCollector(quantized_dtype=quantized_dtype,
                                                    include_layers=calib_layers,
                                                    logger=logger)

        else:
            raise ValueError(f'unknown calibration mode {calib_mode} received,'
                             ' expected `none`, `naive`, or `entropy`')

        num_batches = _collect_layer_statistics(sym_block, calib_data, collector,
                                                len(inputs), num_calib_batches, logger)
        if logger:
            logger.info(f'Collected layer output min/max values from FP32 model using {num_batches} batches')
            logger.info('Performing calibration post collecting operations')

        min_max_dict = collector.post_collect()
        qsym = _calibrate_quantized_sym(qsym, min_max_dict)

    if logger:
        logger.info('Quantizing parameters')
    qarg_params = _quantize_params(qsym, arg_params, min_max_dict)

    if is_np_array():
        qsym = qsym.as_np_ndarray()

    return qsym, qarg_params, aux_params

@wrap_ctx_to_device_func
def quantize_model_onednn(sym, arg_params, aux_params, data_names=('data',),
                          device=cpu(), excluded_sym_names=None, excluded_op_names=None,
                          calib_mode='entropy', calib_data=None, num_calib_batches=None,
                          quantized_dtype='int8', quantize_mode='smart',
                          quantize_granularity='tensor-wise', logger=None):
    """User-level API for generating a fusion + quantized model from a FP32 model
    w/ or w/o calibration with oneDNN.
    The backend quantized operators are only enabled for Linux systems. Please do not run
    inference using the quantized models on Windows for now.

    Parameters
    ----------
    all
        :ref:`As in quantize_model<quantize_model_params>`


    Returns
    -------
    quantized_model: tuple
        A tuple of quantized symbol, quantized arg_params, and aux_params.
    """
    if not isinstance(device, Device):
        raise ValueError(f'currently only supports single device, while received {str(device)}')
    if device.device_type != 'cpu':
        raise ValueError(
            'quantize_model_onednn only support Intel cpu platform with oneDNN Backend')

    sym = sym.optimize_for(backend='ONEDNN_QUANTIZE')

    qsym, qarg_params, aux_params = quantize_model(sym=sym, arg_params=arg_params, aux_params=aux_params,
                                                   data_names=data_names, device=device,
                                                   excluded_sym_names=excluded_sym_names,
                                                   excluded_op_names=excluded_op_names,
                                                   calib_mode=calib_mode, calib_data=calib_data,
                                                   num_calib_batches=num_calib_batches,
                                                   quantized_dtype=quantized_dtype, quantize_mode=quantize_mode,
                                                   quantize_granularity=quantize_granularity, logger=logger)

    qsym = qsym.optimize_for(backend='ONEDNN_QUANTIZE')

    return qsym, qarg_params, aux_params

def quantize_graph(sym, arg_params, aux_params, device=cpu(),
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
    sym : Symbol
        Defines the structure of a neural network for FP32 data types.
    device : Device
        Defines the device that users want to run forward propagation on the calibration
        dataset for collecting layer output statistics. Currently, only supports single device.
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
    quantize_granularity : str
        The granularity of quantization, currently supports 'tensor-wise' and 'channel-wise'
        quantization. The default value is 'tensor-wise'.
    LayerOutputCollector : subclass of CalibrationCollector
        For custom calibration method usage.
        Passed object's include_layers attribute will be feed with names of layers which needs calibration
    logger : Object
        A logging object for printing information during the process of quantization.
    Returns
    -------
    quantized_model : tuple
        A tuple of quantized symbol, quantized arg_params, aux_params and collector.
    """
    if excluded_sym_names is None:
        excluded_sym_names = []
    if not isinstance(excluded_sym_names, list):
        raise ValueError('excluded_sym_names must be a list of strings representing'
                         ' the names of the symbols that will not be quantized,'
                         f' while received type {str(type(excluded_sym_names))}')
    if not isinstance(device, Device):
        raise ValueError(f'currently only supports single device, while received {str(device)}')
    if logger:
        os.environ['MXNET_QUANTIZATION_VERBOSE'] = '1'
        logger.info('Quantizing graph')
    if quantized_dtype not in ('int8', 'uint8', 'auto'):
        raise ValueError(f'unknown quantized_dtype {quantized_dtype} received,'
                         ' expected `int8`, `uint8` or `auto`')
    if quantize_granularity not in ('tensor-wise', 'channel-wise'):
        raise ValueError(f'unkonwn quantize_granularity {quantize_granularity} received,'
                         ' expected `tensor-wise` or `channel-wise`.')
    qsym, calib_layers = _quantize_symbol(sym, device, excluded_symbols=excluded_sym_names,
                                          excluded_operators=excluded_op_names,
                                          offline_params=list(arg_params.keys()),
                                          quantized_dtype=quantized_dtype,
                                          quantize_mode=quantize_mode,
                                          quantize_granularity=quantize_granularity)

    collector = None
    if calib_mode is not None and calib_mode != 'none':
        if calib_mode == 'entropy':
            collector = _LayerHistogramCollector(quantized_dtype=quantized_dtype,
                                                 include_layers=calib_layers, logger=logger)
            if logger:
                logger.info(
                    'Create a layer output collector for entropy calibration.')
        elif calib_mode == 'naive':
            collector = _LayerOutputMinMaxCollector(quantized_dtype=quantized_dtype,
                                                    include_layers=calib_layers, logger=logger)
            if logger:
                logger.info(
                    'Create a layer output minmax collector for naive calibration')
        elif calib_mode == 'custom' and LayerOutputCollector is not None:
            if not isinstance(LayerOutputCollector, CalibrationCollector):
                raise ValueError('LayerOutputCollecotr must be a subclass of a CalibrationCollector class,'
                                 f' but it is {LayerOutputCollector.__class__}')
            collector = LayerOutputCollector

            # Inject layer names that need calibration to collector
            if hasattr(collector, "include_layers"):
                if collector.include_layers is not None:
                    logger.info('Custom collector has set include_layers attribute. '
                                'Calibration layers not passed')
                else:
                    collector.include_layers = calib_layers
            if logger:
                logger.info(
                    'Create a custom layer output minmax collector for calibration')
        else:
            raise ValueError(f'unknown calibration mode {calib_mode} received,'
                             ' expected `none`, `naive`, `entropy` or `custom`')
        if logger:
            logger.info('Collector created, please use set_monitor_callback'
                        ' to collect calibration information.')

    if logger:
        logger.info('Quantizing parameters')
    qarg_params = _quantize_params(qsym, arg_params, min_max_dict={})

    if is_np_array():
        qsym = qsym.as_np_ndarray()

    return qsym, qarg_params, aux_params, collector, calib_layers

def calib_graph(qsym, arg_params, aux_params, collector,
                calib_mode='entropy', logger=None):
    """User-level API for calibrating a quantized model using a filled collector.
    The backend quantized operators are only enabled for Linux systems. Please do not run
    inference using the quantized models on Windows for now.

    Parameters
    ----------
    qsym : Symbol
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
    quantized_model : tuple
        A tuple of calibrated symbol, quantized arg_params, aux_params.
    """
    min_max_dict = {}
    if calib_mode is not None and calib_mode != 'none':
        if calib_mode in ('entropy', 'naive', 'custom'):
            min_max_dict = collector.post_collect()

        else:
            raise ValueError(f'unknown calibration mode {calib_mode} received,'
                             ' expected `none`, `naive`, `entropy` or `custom`')
        qsym = _calibrate_quantized_sym(qsym, min_max_dict)
    else:
        raise ValueError('Please set calibration mode to naive, entropy or custom (with custom CalibrationCollector)')

    if logger:
        logger.info('Quantizing parameters')
    qarg_params = _quantize_params(qsym, arg_params, min_max_dict)

    if is_np_array():
        qsym = qsym.as_np_ndarray()

    return qsym, qarg_params, aux_params

@wrap_ctx_to_device_func
def quantize_net(network, quantized_dtype='auto', quantize_mode='full', quantize_granularity='tensor-wise',
                 exclude_layers=None, exclude_layers_match=None, exclude_operators=None,
                 calib_data=None, data_shapes=None, calib_mode='none',
                 num_calib_batches=None, device=cpu(), LayerOutputCollector=None, logger=None):
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
    calib_data : gluon.DataLoader
        A iterable data loading object.
    data_shapes : list of DataDesc or list of tuple
        A list of data shapes. Required if calib_data is not provided. In case of tuples,
        the names of inputs are generated.
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
        If calib_mode='custom', the provided LayerOutputCollector will be used to determine
        the thresholds for quantization. For more information refer to CalibrationCollector
        documentation.
    num_calib_batches : int or None
        The maximum number of batches that user would like to use for calibration. If not provided,
        the whole calibration dataset will be used.
    device : Device
        Defines the device that users want to run forward propagation on the calibration
        dataset for collecting layer output statistics. Currently, only supports single device.
    LayerOutputCollector : subclass of CalibrationCollector
        For `custom` calibration method usage.
        Passed object's include_layers attribute will be feed with names of layers which needs calibration
    logger : Object
        A logging object for printing information during the process of quantization.

    Returns
    -------
    network : Gluon SymbolBlock
        Defines the structure of a neural network for INT8 data types.
    """
    from ..gluon import SymbolBlock

    if device != mx.cpu():
        raise ValueError('Quantization currently supports only CPU device')
    backend = 'ONEDNN_QUANTIZE'

    network.hybridize(static_alloc=False, static_shape=False)
    data_types = None
    if data_shapes is None:
        if calib_data is None:
            raise ValueError('At least one of data_shapes or calib_data has to be provided.')

        if isinstance(calib_data, mx.gluon.data.DataLoader):
            x = iter(calib_data)
            batch = next(x)
            if isinstance(batch, list):
                data_shapes = _multilist_iterator(batch, lambda x: x.shape)
                data_types = _multilist_iterator(batch, lambda x: x.dtype)
            else:
                data_shapes = [batch.shape]
                data_types = [batch.dtype]
        else:
            raise ValueError('calib_data expects mx.gluon.data.DataLoader')

    if data_types is None:
        data_types = _multilist_iterator(data_shapes, lambda x: mx_real_t)

    data_descs = _generate_list_of_data_desc(data_shapes, data_types)

    num_inputs = len(data_descs)
    data_nd = []
    arr_fn = mx.np if is_np_array() else mx.nd
    data_nd = _multilist_iterator(data_descs, lambda d, F=arr_fn: F.zeros(shape=d.shape, dtype=d.dtype))

    while True:
        try:
            network(*data_nd)
        except (ValueError, TypeError) as err:
            if logger:
                logger.warning(err)
                logger.warning("Deduced input data descriptors failed to run forward pass."
                               " Trying again with one less input.")
            del data_nd[-1]
            num_inputs -= 1
            data_shapes = [b.shape for b in data_nd]
            data_types = [b.dtype for b in data_nd]
            data_descs = _generate_list_of_data_desc(data_shapes, data_types)
            continue
        else:
            break

    symnet, params = network.export(None)
    symnet = symnet.optimize_for(backend=backend)

    if is_np_array():
        symnet = symnet.as_np_ndarray()

    args, auxs = dict(), dict()
    for k, v in params.items():
        ptype, pname = k[:3], k[4:]
        if ptype == "arg":
            args[pname] = v
        else:
            auxs[pname] = v

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
        logger.info(f'These layers have been excluded {exclude_layers}')

    qsym, qarg_params, aux_params, collector, _ = quantize_graph(
        sym=symnet, arg_params=args, aux_params=auxs, device=device,
        excluded_sym_names=exclude_layers, excluded_op_names=exclude_operators,
        calib_mode=calib_mode, quantized_dtype=quantized_dtype, quantize_mode=quantize_mode,
        quantize_granularity=quantize_granularity, LayerOutputCollector=LayerOutputCollector,
        logger=logger)

    if calib_mode is not None and calib_mode != 'none':
        if not isinstance(device, Device):
            raise ValueError(
                f'currently only supports single device, while received {str(device)}')
        if calib_data is None:
            raise ValueError(
                f'calib_data must be provided when calib_mode={calib_mode}')
        if calib_mode in ['naive', 'entropy', 'custom']:
            inputs = _multilist_iterator(data_descs, lambda dd: mx.sym.var(dd.name))
            calib_net = SymbolBlock(symnet, inputs)
            for k, v in calib_net.collect_params().items():
               v.grad_req = 'null'

            calib_net.load_dict(params, cast_dtype=True, dtype_source='saved')
            calib_net.hybridize(static_alloc=False, static_shape=False)
            num_batches = _collect_layer_statistics(calib_net, calib_data, collector, num_inputs,
                                                    num_calib_batches, logger)

            if logger:
                logger.info(f'Collected layer output values from FP32 model using {num_batches} batches')

            qsym, qarg_params, aux_params = calib_graph(
                qsym=qsym, arg_params=args, aux_params=auxs, collector=collector,
                calib_mode=calib_mode, logger=logger)
        else:
            raise ValueError('calib_mode has to be one of: naive, entropy, custom')
    elif calib_mode is not None and calib_mode == 'none':
        inputs = _multilist_iterator(data_descs, lambda dd: mx.sym.var(dd.name))

    net = SymbolBlock(qsym, inputs)
    for k, v in net.collect_params().items():
        v.grad_req = 'null'

    all_params = {(f'arg:{k}'): v.as_in_context(cpu()) for k, v in qarg_params.items()}
    all_params.update({(f'aux:{k}'): v.as_in_context(cpu()) for k, v in aux_params.items()})
    net.load_dict(all_params, cast_dtype=True, dtype_source='saved')
    net.optimize_for(data_nd, backend=backend, skip_infer=True)
    return net
