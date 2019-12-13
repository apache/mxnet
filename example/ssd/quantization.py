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

from __future__ import print_function
import os
import sys
import importlib
import mxnet as mx
from dataset.iterator import DetRecordIter
from config.config import cfg
from evaluate.eval_metric import MApMetric, VOC07MApMetric
import argparse
import logging
import time
from symbol.symbol_factory import get_symbol
from symbol import symbol_builder
from mxnet.base import SymbolHandle, check_call, _LIB, mx_uint, c_str_array
import ctypes
from mxnet.contrib.quantization import *

def save_symbol(fname, sym, logger=None):
    if logger is not None:
        logger.info('Saving symbol into file at %s' % fname)
    sym.save(fname)


def save_params(fname, arg_params, aux_params, logger=None):
    if logger is not None:
        logger.info('Saving params into file at %s' % fname)
    save_dict = {('arg:%s' % k): v.as_in_context(cpu()) for k, v in arg_params.items()}
    save_dict.update({('aux:%s' % k): v.as_in_context(cpu()) for k, v in aux_params.items()})
    mx.nd.save(fname, save_dict)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate a calibrated quantized SSD model from a FP32 model')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--num-calib-batches', type=int, default=5,
                        help='number of batches for calibration')
    parser.add_argument('--exclude-first-conv', action='store_true', default=False,
                        help='excluding quantizing the first conv layer since the'
                             ' number of channels is usually not a multiple of 4 in that layer'
                             ' which does not satisfy the requirement of cuDNN')
    parser.add_argument('--shuffle-dataset', action='store_true', default=True,
                        help='shuffle the calibration dataset')
    parser.add_argument('--shuffle-chunk-seed', type=int, default=3982304,
                        help='shuffling chunk seed, see'
                             ' https://mxnet.apache.org/api/python/io/io.html?highlight=imager#mxnet.io.ImageRecordIter'
                             ' for more details')
    parser.add_argument('--shuffle-seed', type=int, default=48564309,
                        help='shuffling seed, see'
                             ' https://mxnet.apache.org/api/python/io/io.html?highlight=imager#mxnet.io.ImageRecordIter'
                             ' for more details')
    parser.add_argument('--calib-mode', type=str, default='naive',
                        help='calibration mode used for generating calibration table for the quantized symbol; supports'
                             ' 1. none: no calibration will be used. The thresholds for quantization will be calculated'
                             ' on the fly. This will result in inference speed slowdown and loss of accuracy'
                             ' in general.'
                             ' 2. naive: simply take min and max values of layer outputs as thresholds for'
                             ' quantization. In general, the inference accuracy worsens with more examples used in'
                             ' calibration. It is recommended to use `entropy` mode as it produces more accurate'
                             ' inference results.'
                             ' 3. entropy: calculate KL divergence of the fp32 output and quantized output for optimal'
                             ' thresholds. This mode is expected to produce the best inference accuracy of all three'
                             ' kinds of quantized models if the calibration dataset is representative enough of the'
                             ' inference dataset.')
    parser.add_argument('--quantized-dtype', type=str, default='auto',
                        choices=['auto', 'int8', 'uint8'],
                        help='quantization destination data type for input data')

    args = parser.parse_args()
    ctx = mx.cpu(0)
    logging.basicConfig()
    logger = logging.getLogger('logger')
    logger.setLevel(logging.INFO)

    logger.info('shuffle_dataset=%s' % args.shuffle_dataset)

    calib_mode = args.calib_mode
    logger.info('calibration mode set to %s' % calib_mode)

    # load FP32 models
    prefix, epoch = "./model/ssd_vgg16_reduced_300", 0
    sym, arg_params, aux_params = mx.model.load_checkpoint("./model/ssd_vgg16_reduced_300", 0)

    if not 'label' in sym.list_arguments():
        label = mx.sym.Variable(name='label')
        sym = mx.sym.Group([sym, label])

    sym = sym.get_backend_symbol('MKLDNN_QUANTIZE')

    # get batch size
    batch_size = args.batch_size
    logger.info('batch size = %d for calibration' % batch_size)

    # get number of batches for calibration
    num_calib_batches = args.num_calib_batches
    if calib_mode != 'none':
        logger.info('number of batches = %d for calibration' % num_calib_batches)

    # get image shape
    image_shape = '3,300,300'

    # Quantization layer configs
    exclude_first_conv = args.exclude_first_conv
    excluded_sym_names = []
    rgb_mean = '123,117,104'
    if exclude_first_conv:
        excluded_sym_names += ['conv1_1']

    label_name = 'label'
    logger.info('label_name = %s' % label_name)

    data_shape = tuple([int(i) for i in image_shape.split(',')])
    logger.info('Input data shape = %s' % str(data_shape))

    logger.info('rgb_mean = %s' % rgb_mean)
    rgb_mean = [float(i) for i in rgb_mean.split(',')]
    mean_args = {'mean_r': rgb_mean[0], 'mean_g': rgb_mean[1], 'mean_b': rgb_mean[2]}

    if calib_mode == 'none':
        qsym, qarg_params, aux_params = quantize_model(sym=sym, arg_params=arg_params, aux_params=aux_params,
                                                       ctx=ctx, excluded_sym_names=excluded_sym_names,
                                                       calib_mode=calib_mode, quantized_dtype=args.quantized_dtype,
                                                       logger=logger)
        sym_name = '%s-symbol.json' % ('./model/qssd_vgg16_reduced_300')
        param_name = '%s-%04d.params' % ('./model/qssd_vgg16_reduced_300', epoch)
        save_symbol(sym_name, qsym, logger)
    else:
        logger.info('Creating ImageRecordIter for reading calibration dataset')
        eval_iter = DetRecordIter(os.path.join(os.getcwd(), 'data', 'val.rec'),
                                  batch_size, data_shape, mean_pixels=(123, 117, 104),
                                  path_imglist="", **cfg.valid)

        qsym, qarg_params, aux_params = quantize_model(sym=sym, arg_params=arg_params, aux_params=aux_params,
                                                        ctx=ctx, excluded_sym_names=excluded_sym_names,
                                                        calib_mode=calib_mode, calib_data=eval_iter,
                                                        num_calib_examples=num_calib_batches * batch_size,
                                                        quantized_dtype=args.quantized_dtype,
                                                        label_names=(label_name,), logger=logger)
        sym_name = '%s-symbol.json' % ('./model/cqssd_vgg16_reduced_300')
        param_name = '%s-%04d.params' % ('./model/cqssd_vgg16_reduced_300', epoch)
    qsym = qsym.get_backend_symbol('MKLDNN_QUANTIZE')
    save_symbol(sym_name, qsym, logger)
    save_params(param_name, qarg_params, aux_params, logger)
