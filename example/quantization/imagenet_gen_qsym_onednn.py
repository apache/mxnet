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
import logging
import os
import re
import sys
from inspect import currentframe, getframeinfo

import mxnet as mx
from mxnet import gluon
from mxnet.contrib.quantization import quantize_net
from mxnet.gluon.data import DataLoader
from mxnet.gluon.data.vision import transforms
from mxnet.gluon.model_zoo.vision import get_model

SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, '..', '..')))
from tools.rec2idx import IndexCreator


def download_calib_dataset(dataset_url, calib_dataset, logger=None):
    if logger is not None:
        logger.info(f'Downloading calibration dataset from {dataset_url} to {calib_dataset}')
    mx.test_utils.download(dataset_url, calib_dataset)


def get_from_gluon(model_name, classes=1000, logger=None):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    model_path = os.path.join(dir_path, 'model')
    if logger is not None:
        logger.info(f'Converting model from Gluon-CV ModelZoo {model_name}... into path {model_path}')
    net = get_model(name=model_name, classes=classes, pretrained=True)
    prefix = os.path.join(model_path, model_name)
    return net, prefix


def regex_find_excluded_symbols(patterns_dict, model_name):
    for key, value in patterns_dict.items():
        if re.search(key, model_name) is not None:
            return value
    return None


def get_exclude_symbols(model_name, exclude_first_conv):
    """Grouped supported models at the time of commit:
    - alexnet
    - densenet121, densenet161
    - densenet169, densenet201
    - inceptionv3
    - mobilenet0.25, mobilenet0.5, mobilenet0.75, mobilenet1.0,
    - mobilenetv2_0.25, mobilenetv2_0.5, mobilenetv2_0.75, mobilenetv2_1.0
    - resnet101_v1, resnet152_v1, resnet18_v1, resnet34_v1, resnet50_v1
    - resnet101_v2, resnet152_v2, resnet18_v2, resnet34_v2, resnet50_v2
    - squeezenet1.0, squeezenet1.1
    - vgg11, vgg11_bn, vgg13, vgg13_bn, vgg16, vgg16_bn, vgg19, vgg19_bn
    """
    exclude_symbol_regex = {
        'mobilenet[^v]': ['mobilenet_hybridsequential0_flatten0_flatten0', 'mobilenet_hybridsequential0_globalavgpool2d0_fwd'],
        'mobilenetv2': ['mobilenetv2_hybridsequential1_flatten0_flatten0'],
        # resnetv2_hybridsequential0_hybridsequential0_bottleneckv20_batchnorm0_fwd is excluded for the sake of accuracy
        'resnet.*v2': ['resnetv2_hybridsequential0_flatten0_flatten0', 'resnetv2_hybridsequential0_hybridsequential0_bottleneckv20_batchnorm0_fwd'],
        'squeezenet1': ['squeezenet_hybridsequential1_flatten0_flatten0'],
    }
    excluded_sym_names = regex_find_excluded_symbols(exclude_symbol_regex, model_name)
    if excluded_sym_names is None:
        excluded_sym_names = []
    if exclude_first_conv:
        first_conv_regex = {
            'alexnet': ['alexnet_hybridsequential0_conv2d0_fwd'],
            'densenet': ['densenet_hybridsequential0_conv2d0_fwd'],
            'inceptionv3': ['inception3_hybridsequential0_hybridsequential0_conv2d0_fwd'],
            'mobilenet[^v]': ['mobilenet_hybridsequential0_conv2d0_fwd'],
            'mobilenetv2': ['mobilenetv2_hybridsequential0_conv2d0_fwd'],
            'resnet.*v1': ['resnetv1_hybridsequential0_conv2d0_fwd'],
            'resnet.*v2': ['resnetv2_hybridsequential0_conv2d0_fwd'],
            'squeezenet1': ['squeezenet_hybridsequential0_conv2d0_fwd'],
            'vgg': ['vgg_hybridsequential0_conv2d0_fwd'],
        }
        excluded_first_conv_sym_names = regex_find_excluded_symbols(first_conv_regex, model_name)
        if excluded_first_conv_sym_names is None:
            raise ValueError(f'Currently, model {model_name} is not supported in this script')
        excluded_sym_names += excluded_first_conv_sym_names
    return excluded_sym_names


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate a calibrated quantized model from a FP32 model with oneDNN support')
    parser.add_argument('--model', type=str, default='resnet50_v1',
                        help='model to be quantized. If no-pretrained is set then'
                             'model must be provided to `model` directory in the same path'
                             'as this python script')
    parser.add_argument('--epoch', type=int, default=0,
                        help='number of epochs, default is 0')
    parser.add_argument('--no-pretrained', action='store_true', default=False,
                        help='If enabled, will not download pretrained model from MXNet or Gluon-CV modelzoo.')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--calib-dataset', type=str, default='data/val_256_q90.rec',
                        help='path of the calibration dataset')
    parser.add_argument('--image-shape', type=str, default='3,224,224',
                        help='number of channels, height and width of input image separated by comma')
    parser.add_argument('--data-nthreads', type=int, default=0,
                        help='number of threads for data loading')
    parser.add_argument('--num-calib-batches', type=int, default=10,
                        help='number of batches for calibration')
    parser.add_argument('--exclude-first-conv', action='store_true', default=False,
                        help='excluding quantizing the first conv layer since the'
                             ' input data may have negative value which doesn\'t support at moment')
    parser.add_argument('--shuffle-dataset', action='store_true',
                        help='shuffle the calibration dataset')
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
                             ' kinds of calibration modes if the calibration dataset is representative enough of the'
                             ' inference dataset.')
    parser.add_argument('--quantized-dtype', type=str, default='auto',
                        choices=['auto', 'int8', 'uint8'],
                        help='quantization destination data type for input data')
    parser.add_argument('--quiet', action='store_true', default=False,
                        help='suppress most of log')
    args = parser.parse_args()
    ctx = mx.cpu(0)
    logger = None

    if not args.quiet:
        logging.basicConfig()
        logger = logging.getLogger('logger')
        logger.setLevel(logging.INFO)

    if logger:
        logger.info(args)
        logger.info(f'shuffle_dataset={args.shuffle_dataset}')
        logger.info(f'calibration mode set to {args.calib_mode}')

    calib_mode = args.calib_mode

    # download calibration dataset
    if calib_mode != 'none':
        idx_file_name = os.path.splitext(args.calib_dataset)[0] + '.idx'
        if not os.path.isfile(idx_file_name):
            download_calib_dataset('http://data.mxnet.io/data/val_256_q90.rec', args.calib_dataset)
            creator = IndexCreator(args.calib_dataset, idx_file_name)
            creator.create_index()
            creator.close()

    # get image shape
    image_shape = args.image_shape
    data_shape = [(1,) + tuple(int(i) for i in image_shape.split(','))]

    # check if directory for output model exists
    dir_path = os.path.dirname(os.path.realpath(__file__))
    dir_path = os.path.join(dir_path, 'model')
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)  # without try catch block as we expect to finish script if it fail

    # download model
    if not args.no_pretrained:
        if logger:
            logger.info('Get pre-trained model from Gluon-CV modelzoo.')
            logger.info('If you want to use custom model, please set --no-pretrained.')
        net, prefix = get_from_gluon(model_name=args.model, classes=1000, logger=logger)
        rgb_mean = '0.485,0.456,0.406'
        rgb_std = '0.229,0.224,0.225'
        epoch = 0
        net.hybridize()
        net(mx.np.zeros(data_shape[0])) # dummy forward pass to build graph
        net.export(prefix) # save model
        net.hybridize(active=False) # disable hybridization - it will be handled in quantization API
    else:
        prefix = os.path.join(dir_path, args.model)
        epoch = args.epoch
        net = gluon.SymbolBlock.imports("{}-symbol.json".format(prefix), ['data'], "{}-0000.params".format(prefix))

    # get batch size
    batch_size = args.batch_size
    if logger:
        logger.info(f'batch size = {batch_size} for calibration')

    # get number of batches for calibration
    num_calib_batches = args.num_calib_batches
    if logger:
        if calib_mode == 'none':
            logger.info('skip calibration step as calib_mode is none')
        else:
            logger.info(f'number of batches = {num_calib_batches} for calibration')

    # get number of threads for decoding the dataset
    data_nthreads = args.data_nthreads

    exclude_first_conv = args.exclude_first_conv
    if args.quantized_dtype == "uint8":
        if logger:
            logger.info('quantized dtype is set to uint8, will exclude first conv.')
        exclude_first_conv = True
    excluded_sym_names = []
    if not args.no_pretrained:
        excluded_sym_names += get_exclude_symbols(args.model, args.exclude_first_conv)
    else:
        if logger:
            frameinfo = getframeinfo(currentframe())
            logger.info(F'Please set proper RGB configs inside this script below {frameinfo.filename}:{frameinfo.lineno} for model {args.model}!')
        # add rgb mean/std of your model.
        rgb_mean = '0,0,0'
        rgb_std = '0,0,0'
        # add layer names you donnot want to quantize.
        if logger:
            frameinfo = getframeinfo(currentframe())
            logger.info(F'Please set proper excluded_sym_names inside this script below {frameinfo.filename}:{frameinfo.lineno} for model {args.model} if required!')
        excluded_sym_names += []
        if exclude_first_conv:
            excluded_sym_names += []

    if logger:
        logger.info(f'These layers have been excluded {excluded_sym_names}')
        logger.info(f'Input data shape = {str(data_shape)}')
        logger.info(f'rgb_mean = {rgb_mean}')
        logger.info(f'rgb_std = {rgb_std}')

    rgb_mean = [float(i) for i in rgb_mean.split(',')]
    mean_args = {'mean_r': rgb_mean[0], 'mean_g': rgb_mean[1], 'mean_b': rgb_mean[2]}
    rgb_std = [float(i) for i in rgb_std.split(',')]
    std_args = {'std_r': rgb_std[0], 'std_g': rgb_std[1], 'std_b': rgb_std[2]}
    if calib_mode == 'none':
        if logger:
            logger.info(f'Quantizing FP32 model {args.model}')
        qsym = quantize_net(net, ctx=ctx, exclude_layers_match=excluded_sym_names, data_shapes=data_shape,
                            calib_mode=calib_mode, quantized_dtype=args.quantized_dtype,
                            logger=logger)
        suffix = '-quantized'
    else:
        if logger:
            logger.info('Creating DataLoader for reading calibration dataset')
        dataset = mx.gluon.data.vision.ImageRecordDataset(args.calib_dataset)
        transformer = transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=rgb_mean, std=rgb_std)])
        data_loader = DataLoader(dataset.transform_first(transformer), batch_size, shuffle=args.shuffle_dataset, num_workers=data_nthreads)
        qsym = quantize_net(net, ctx=ctx, exclude_layers_match=excluded_sym_names,
                            calib_mode=calib_mode, calib_data=data_loader, num_calib_batches=num_calib_batches,
                            quantized_dtype=args.quantized_dtype, logger=logger)
        if calib_mode == 'entropy':
            suffix = f'-quantized-{num_calib_batches}batches-entropy'
        elif calib_mode == 'naive':
            suffix = f'-quantized-{num_calib_batches}batches-naive'
        else:
            raise ValueError(f'unknown calibration mode {calib_mode} received, only supports `none`, `naive`, and `entropy`')
    save_path = prefix + suffix
    model_path, params_path = qsym.export(save_path, epoch)
    if logger is not None:
        logger.info(F'Saved quantized model into:\n{model_path}\n{params_path}')
