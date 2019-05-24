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
import gluoncv
from mxnet import gluon, nd, image
from gluoncv import utils
from gluoncv.model_zoo import get_model
from mxnet.contrib.quantization import *
from mxnet.base import SymbolHandle, check_call, _LIB, mx_uint, c_str_array
import ctypes


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

def convert_from_gluon(model_name, image_shape, classes=1000, logger=None):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    model_path = os.path.join(dir_path, 'model')
    if logger is not None:
        logger.info('Converting model from Gluon-CV ModelZoo %s... into path %s' % (model_name, model_path))
    net = get_model(name=model_name, classes=classes, pretrained=True)
    net.hybridize()
    x = mx.sym.var('data')
    y = net(x)
    y = mx.sym.SoftmaxOutput(data=y, name='softmax')
    symnet = mx.symbol.load_json(y.tojson())
    params = net.collect_params()
    args = {}
    auxs = {}
    for param in params.values():
        v = param._reduce()
        k = param.name
        if 'running' in k:
            auxs[k] = v
        else:
            args[k] = v
    mod = mx.mod.Module(symbol=symnet, context=mx.cpu(),
                        label_names = ['softmax_label'])
    mod.bind(for_training=False,
             data_shapes=[('data', (1,) +
                          tuple([int(i) for i in image_shape.split(',')]))])
    mod.set_params(arg_params=args, aux_params=auxs)
    dst_dir = os.path.join(dir_path, 'model')
    prefix = os.path.join(dir_path, 'model', model_name)
    if not os.path.isdir(dst_dir):
        os.mkdir(dst_dir)
    mod.save_checkpoint(prefix, 0)
    return prefix

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
    parser = argparse.ArgumentParser(description='Generate a calibrated quantized model from a FP32 model with Intel MKL-DNN support')
    parser.add_argument('--model', type=str, choices=['resnet18_v1',
                                                      'resnet50_v1',
                                                      'resnet101_v1',
                                                      'inceptionv3',
                                                      'squeezenet1.0',
                                                      'mobilenet1.0',
                                                      'mobilenetv2_1.0',
                                                      'imagenet1k-resnet-152',
                                                      'imagenet1k-inception-bn',
                                                      'custom'],
                        help='currently only supports imagenet1k-resnet-50_v1, imagenet1k-resnet-152 or imagenet1k-inception-bn.'
                             'you can set to custom to load your pre-trained model.')
    parser.add_argument('--use-gluon-model', type=bool, default=False,
                        help='If enabled, will download pretrained model from Gluon-CV '
                             'and convert to symbolic model ')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--label-name', type=str, default='softmax_label')
    parser.add_argument('--calib-dataset', type=str, default='data/val_256_q90.rec',
                        help='path of the calibration dataset')
    parser.add_argument('--image-shape', type=str, default='3,224,224')
    parser.add_argument('--data-nthreads', type=int, default=60,
                        help='number of threads for data decoding')
    parser.add_argument('--num-calib-batches', type=int, default=10,
                        help='number of batches for calibration')
    parser.add_argument('--exclude-first-conv', action='store_true', default=False,
                        help='excluding quantizing the first conv layer since the'
                             ' input data may have negative value which doesn\'t support at moment' )
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
    parser.add_argument('--quantized-dtype', type=str, default='auto',
                        choices=['auto', 'int8', 'uint8'],
                        help='quantization destination data type for input data')
    parser.add_argument('--enable-calib-quantize', type=bool, default=True,
                        help='If enabled, the quantize op will '
                             'be calibrated offline if calibration mode is '
                             'enabled')
    args = parser.parse_args()
    ctx = mx.cpu(0)
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
    if args.model in ['resnet18_v1',
                      'resnet50_v1',
                      'resnet101_v1',
                      'squeezenet1.0',
                      'mobilenet1.0',
                      'mobilenetv2_1.0',
                      'inceptionv3']:
        logger.info('model %s is converted from GluonCV' % args.model)
        args.use_gluon_model = True
    if args.use_gluon_model == True:
        prefix = convert_from_gluon(model_name=args.model, image_shape=args.image_shape, classes=1000, logger=logger)
        epoch = 0
        sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, epoch)
    elif args.model == 'custom':
        dir_path = os.path.dirname(os.path.realpath(__file__))
        prefix = os.path.join(dir_path, 'model', args.model)
        epoch = 0
        sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, epoch)
    else:
        prefix, epoch = download_model(model_name=args.model, logger=logger)
        sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, epoch)

    sym = sym.get_backend_symbol('MKLDNN_QUANTIZE')

    # get batch size
    batch_size = args.batch_size
    logger.info('batch size = %d for calibration' % batch_size)

    # get number of batches for calibration
    num_calib_batches = args.num_calib_batches
    if calib_mode == 'none':
        logger.info('skip calibration step as calib_mode is none')
    else:
        logger.info('number of batches = %d for calibration' % num_calib_batches)

    # get number of threads for decoding the dataset
    data_nthreads = args.data_nthreads

    # get image shape
    image_shape = args.image_shape

    calib_layer = lambda name: name.endswith('_output') or name == "data"
    exclude_first_conv = args.exclude_first_conv
    if args.quantized_dtype == "uint8":
        logger.info('quantized dtype is set to uint8, will exclude first conv.')
        exclude_first_conv = True
    excluded_sym_names = []
    if args.model == 'imagenet1k-resnet-152':
        rgb_mean = '0,0,0'
        rgb_std = '1,1,1'
        excluded_sym_names += ['flatten0']
        if exclude_first_conv:
            excluded_sym_names += ['conv0']
    elif args.model == 'imagenet1k-inception-bn':
        rgb_mean = '123.68,116.779,103.939'
        rgb_std = '1,1,1'
        excluded_sym_names += ['flatten']
        if exclude_first_conv:
            excluded_sym_names += ['conv_1']
    elif args.model in ['resnet18_v1', 'resnet50_v1', 'resnet101_v1']:
        rgb_mean = '123.68,116.779,103.939'
        rgb_std = '58.393, 57.12, 57.375'
        if exclude_first_conv:
            excluded_sym_names += ['resnetv10_conv0_fwd']
    elif args.model == 'squeezenet1.0':
        rgb_mean = '123.68,116.779,103.939'
        rgb_std = '58.393, 57.12, 57.375'
        excluded_sym_names += ['squeezenet0_flatten0_flatten0']
        if exclude_first_conv:
            excluded_sym_names += ['squeezenet0_conv0_fwd']
    elif args.model == 'mobilenet1.0':
        rgb_mean = '123.68,116.779,103.939'
        rgb_std = '58.393, 57.12, 57.375'
        excluded_sym_names += ['mobilenet0_flatten0_flatten0',
                               'mobilenet0_pool0_fwd']
        if exclude_first_conv:
            excluded_sym_names += ['mobilenet0_conv0_fwd']
    elif args.model == 'mobilenetv2_1.0':
        rgb_mean = '123.68,116.779,103.939'
        rgb_std = '58.393, 57.12, 57.375'
        excluded_sym_names += ['mobilenetv20_output_flatten0_flatten0']
        if exclude_first_conv:
            excluded_sym_names += ['mobilenetv20_conv0_fwd']
    elif args.model == 'inceptionv3':
        rgb_mean = '123.68,116.779,103.939'
        rgb_std = '58.393, 57.12, 57.375'
        if exclude_first_conv:
            excluded_sym_names += ['inception30_conv0_fwd']
    elif args.model == 'custom':
        # add rgb mean/std of your model.
        rgb_mean = '0,0,0'
        rgb_std = '0,0,0'
        # add layer names you donnot want to quantize.
        excluded_sym_names += ['layers']
        if exclude_first_conv:
            excluded_sym_names += ['layers']
    else:
        raise ValueError('model %s is not supported in this script' % args.model)

    label_name = args.label_name
    logger.info('label_name = %s' % label_name)

    data_shape = tuple([int(i) for i in image_shape.split(',')])
    logger.info('Input data shape = %s' % str(data_shape))

    logger.info('rgb_mean = %s' % rgb_mean)
    rgb_mean = [float(i) for i in rgb_mean.split(',')]
    mean_args = {'mean_r': rgb_mean[0], 'mean_g': rgb_mean[1], 'mean_b': rgb_mean[2]}
    logger.info('rgb_std = %s' % rgb_std)
    rgb_std = [float(i) for i in rgb_std.split(',')]
    std_args = {'std_r': rgb_std[0], 'std_g': rgb_std[1], 'std_b': rgb_std[2]}
    combine_mean_std = {}
    combine_mean_std.update(mean_args)
    combine_mean_std.update(std_args)
    if calib_mode == 'none':
        logger.info('Quantizing FP32 model %s' % args.model)
        qsym, qarg_params, aux_params = quantize_model(sym=sym, arg_params=arg_params, aux_params=aux_params,
                                                       ctx=ctx, excluded_sym_names=excluded_sym_names,
                                                       calib_mode=calib_mode, quantized_dtype=args.quantized_dtype,
                                                       logger=logger)
        sym_name = '%s-symbol.json' % (prefix + '-quantized')
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
                                     **combine_mean_std)

        qsym, qarg_params, aux_params = quantize_model(sym=sym, arg_params=arg_params, aux_params=aux_params,
                                                        ctx=ctx, excluded_sym_names=excluded_sym_names,
                                                        calib_mode=calib_mode, calib_data=data,
                                                        num_calib_examples=num_calib_batches * batch_size,
                                                        calib_layer=calib_layer, quantized_dtype=args.quantized_dtype,
                                                        label_names=(label_name,), logger=logger)
        if calib_mode == 'entropy':
            suffix = '-quantized-%dbatches-entropy' % num_calib_batches
        elif calib_mode == 'naive':
            suffix = '-quantized-%dbatches-naive' % num_calib_batches
        else:
            raise ValueError('unknow calibration mode %s received, only supports `none`, `naive`, and `entropy`'
                             % calib_mode)
        sym_name = '%s-symbol.json' % (prefix + suffix)
    qsym = qsym.get_backend_symbol('MKLDNN_QUANTIZE')
    save_symbol(sym_name, qsym, logger)
    param_name = '%s-%04d.params' % (prefix + '-quantized', epoch)
    save_params(param_name, qarg_params, aux_params, logger)
