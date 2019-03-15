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

import argparse
import os
import logging
from common import modelzoo
import mxnet as mx
from mxnet.contrib.quantization import *


def download_calib_dataset(dataset_url, calib_dataset, logger=None):
    if logger is not None:
        logger.info('Downloading calibration dataset from %s to %s' % (dataset_url, calib_dataset))
    mx.test_utils.download(dataset_url, calib_dataset)


def download_model(model_name, logger=None):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    model_path = os.path.join(dir_path, 'model')
    if logger is not None:
        logger.info('Downloading model %s... into path %s' % (model_name, model_path))
    return modelzoo.download_model(args.model, os.path.join(dir_path, 'model'))


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
    parser = argparse.ArgumentParser(description='Generate a calibrated quantized model from a FP32 model')
    parser.add_argument('--ctx', type=str, default='gpu')
    parser.add_argument('--model', type=str, choices=['imagenet1k-resnet-152', 'imagenet1k-inception-bn'],
                        help='currently only supports imagenet1k-resnet-152 or imagenet1k-inception-bn')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--label-name', type=str, default='softmax_label')
    parser.add_argument('--calib-dataset', type=str, default='data/val_256_q90.rec',
                        help='path of the calibration dataset')
    parser.add_argument('--image-shape', type=str, default='3,224,224')
    parser.add_argument('--data-nthreads', type=int, default=60,
                        help='number of threads for data decoding')
    parser.add_argument('--num-calib-batches', type=int, default=10,
                        help='number of batches for calibration')
    parser.add_argument('--exclude-first-conv', action='store_true', default=True,
                        help='excluding quantizing the first conv layer since the'
                             ' number of channels is usually not a multiple of 4 in that layer'
                             ' which does not satisfy the requirement of cuDNN')
    parser.add_argument('--shuffle-dataset', action='store_true', default=True,
                        help='shuffle the calibration dataset')
    parser.add_argument('--shuffle-chunk-seed', type=int, default=3982304,
                        help='shuffling chunk seed, see'
                             ' https://mxnet.incubator.apache.org/api/python/io/io.html?highlight=imager#mxnet.io.ImageRecordIter'
                             ' for more details')
    parser.add_argument('--shuffle-seed', type=int, default=48564309,
                        help='shuffling seed, see'
                             ' https://mxnet.incubator.apache.org/api/python/io/io.html?highlight=imager#mxnet.io.ImageRecordIter'
                             ' for more details')
    parser.add_argument('--calib-mode', type=str, default='entropy',
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
    parser.add_argument('--quantized-dtype', type=str, default='int8',
                        choices=['int8', 'uint8'],
                        help='quantization destination data type for input data')
    args = parser.parse_args()

    if args.ctx == 'gpu':
        ctx = mx.gpu(0)
    elif args.ctx == 'cpu':
        ctx = mx.cpu(0)
    else:
        raise ValueError('ctx %s is not supported in this script' % args.ctx)

    logging.basicConfig()
    logger = logging.getLogger('logger')
    logger.setLevel(logging.INFO)

    logger.info('shuffle_dataset=%s' % args.shuffle_dataset)

    calib_mode = args.calib_mode
    logger.info('calibration mode set to %s' % calib_mode)

    # download calibration dataset
    if calib_mode != 'none':
        download_calib_dataset('http://data.mxnet.io/data/val_256_q90.rec', args.calib_dataset)

    # download model
    prefix, epoch = download_model(model_name=args.model, logger=logger)
    sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, epoch)

    # get batch size
    batch_size = args.batch_size
    logger.info('batch size = %d for calibration' % batch_size)

    # get number of batches for calibration
    num_calib_batches = args.num_calib_batches
    if calib_mode != 'none':
        logger.info('number of batches = %d for calibration' % num_calib_batches)

    # get number of threads for decoding the dataset
    data_nthreads = args.data_nthreads

    # get image shape
    image_shape = args.image_shape

    exclude_first_conv = args.exclude_first_conv
    excluded_sym_names = []
    if args.model == 'imagenet1k-resnet-152':
        rgb_mean = '0,0,0'
        if args.ctx == 'gpu':
            calib_layer = lambda name: name.endswith('_output') and (name.find('conv') != -1
                                                                     or name.find('sc') != -1
                                                                     or name.find('fc') != -1)
        else:
            calib_layer = lambda name: name.endswith('_output') and (name.find('conv') != -1
                                                                     or name.find('sc') != -1)
            excluded_sym_names += ['flatten0', 'fc1']
        if exclude_first_conv:
            excluded_sym_names += ['conv0']
    elif args.model == 'imagenet1k-inception-bn':
        rgb_mean = '123.68,116.779,103.939'
        if args.ctx == 'gpu':
            calib_layer = lambda name: name.endswith('_output') and (name.find('conv') != -1
                                                                     or name.find('fc') != -1)
            excluded_sym_names += ['ch_concat_3a_chconcat',
                                   'ch_concat_3b_chconcat',
                                   'ch_concat_3c_chconcat',
                                   'ch_concat_4a_chconcat',
                                   'ch_concat_4b_chconcat',
                                   'ch_concat_4c_chconcat',
                                   'ch_concat_4d_chconcat',
                                   'ch_concat_4e_chconcat',
                                   'ch_concat_5a_chconcat',
                                   'ch_concat_5b_chconcat']
        else:
            calib_layer = lambda name: name.endswith('_output') and (name.find('conv') != -1)
            excluded_sym_names += ['flatten', 'fc1']
        if exclude_first_conv:
            excluded_sym_names += ['conv_1']
    else:
        raise ValueError('model %s is not supported in this script' % args.model)

    label_name = args.label_name
    logger.info('label_name = %s' % label_name)

    data_shape = tuple([int(i) for i in image_shape.split(',')])
    logger.info('Input data shape = %s' % str(data_shape))

    logger.info('rgb_mean = %s' % rgb_mean)
    rgb_mean = [float(i) for i in rgb_mean.split(',')]
    mean_args = {'mean_r': rgb_mean[0], 'mean_g': rgb_mean[1], 'mean_b': rgb_mean[2]}

    if calib_mode == 'none':
        logger.info('Quantizing FP32 model %s' % args.model)
        qsym, qarg_params, aux_params = quantize_model(sym=sym, arg_params=arg_params, aux_params=aux_params,
                                                       ctx=ctx, excluded_sym_names=excluded_sym_names,
                                                       calib_mode=calib_mode, quantized_dtype=args.quantized_dtype,
                                                       logger=logger)
        sym_name = '%s-symbol.json' % (prefix + '-quantized')
        save_symbol(sym_name, qsym, logger)
    else:
        logger.info('Creating ImageRecordIter for reading calibration dataset')
        data = mx.io.ImageRecordIter(path_imgrec=args.calib_dataset,
                                     label_width=1,
                                     preprocess_threads=data_nthreads,
                                     batch_size=batch_size,
                                     data_shape=data_shape,
                                     label_name=label_name,
                                     rand_crop=False,
                                     rand_mirror=False,
                                     shuffle=args.shuffle_dataset,
                                     shuffle_chunk_seed=args.shuffle_chunk_seed,
                                     seed=args.shuffle_seed,
                                     **mean_args)

        cqsym, qarg_params, aux_params = quantize_model(sym=sym, arg_params=arg_params, aux_params=aux_params,
                                                        ctx=ctx, excluded_sym_names=excluded_sym_names,
                                                        calib_mode=calib_mode, calib_data=data,
                                                        num_calib_examples=num_calib_batches * batch_size,
                                                        calib_layer=calib_layer, quantized_dtype=args.quantized_dtype,
                                                        logger=logger)
        if calib_mode == 'entropy':
            suffix = '-quantized-%dbatches-entropy' % num_calib_batches
        elif calib_mode == 'naive':
            suffix = '-quantized-%dbatches-naive' % num_calib_batches
        else:
            raise ValueError('unknow calibration mode %s received, only supports `none`, `naive`, and `entropy`'
                             % calib_mode)
        sym_name = '%s-symbol.json' % (prefix + suffix)
        save_symbol(sym_name, cqsym, logger)

    param_name = '%s-%04d.params' % (prefix + '-quantized', epoch)
    save_params(param_name, qarg_params, aux_params, logger)
