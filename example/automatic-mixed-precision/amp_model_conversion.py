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

import os
import logging
import argparse
import mxnet as mx
from common import modelzoo
import gluoncv
from gluoncv.model_zoo import get_model
from mxnet import amp
import numpy as np


def save_symbol(fname, sym, logger=None):
    if logger is not None:
        logger.info('Saving symbol into file at {}'.format(fname))
    sym.save(fname, remove_amp_cast=False)


def save_params(fname, arg_params, aux_params, logger=None):
    if logger is not None:
        logger.info('Saving params into file at {}'.format(fname))
    save_dict = {('arg:%s' % k): v.as_in_context(mx.cpu()) for k, v in arg_params.items()}
    save_dict.update({('aux:%s' % k): v.as_in_context(mx.cpu()) for k, v in aux_params.items()})
    mx.nd.save(fname, save_dict)


if __name__ == '__main__':
    # Faster RCNN and Mask RCNN commented because of model loading issues
    # https://github.com/dmlc/gluon-cv/issues/1034
    gluon_models = [#'faster_rcnn_fpn_resnet50_v1b_coco',
                    'mobilenetv2_0.75',
                    'cifar_resnet56_v1',
                    'mobilenet0.25',
                    'mobilenet1.0',
                    #'mask_rcnn_fpn_resnet50_v1b_coco',
                    'simple_pose_resnet152_v1b',
                    'ssd_512_resnet50_v1_voc',
                    #'faster_rcnn_resnet50_v1b_voc',
                    'cifar_resnet20_v1',
                    'yolo3_darknet53_voc',
                    'resnet101_v1c',
                    'simple_pose_resnet18_v1b',
                    #'mask_rcnn_resnet50_v1b_coco',
                    'ssd_512_mobilenet1.0_coco',
                    'vgg19_bn',
                    #'faster_rcnn_resnet50_v1b_coco',
                    'cifar_resnet110_v1',
                    'yolo3_mobilenet1.0_voc',
                    'cifar_resnext29_16x64d',
                    'resnet34_v1',
                    'densenet121',
                    #'mask_rcnn_fpn_resnet101_v1d_coco',
                    'vgg13_bn',
                    'vgg19',
                    'resnet152_v1d',
                    'resnet152_v1s',
                    'densenet201',
                    'alexnet',
                    'se_resnext50_32x4d',
                    'resnet50_v1d_0.86',
                    'resnet18_v1b_0.89',
                    'yolo3_darknet53_coco',
                    'resnet152_v1',
                    'resnext101_64x4d',
                    'vgg13',
                    'resnet101_v1d_0.76',
                    'simple_pose_resnet50_v1d',
                    'senet_154',
                    'resnet50_v1',
                    'se_resnext101_32x4d',
                    'fcn_resnet101_voc',
                    'resnet152_v2',
                    #'mask_rcnn_resnet101_v1d_coco',
                    'squeezenet1.1',
                    'mobilenet0.5',
                    'resnet34_v2',
                    'resnet18_v1',
                    'resnet152_v1b',
                    'resnet101_v2',
                    'cifar_resnet56_v2',
                    'ssd_512_resnet101_v2_voc',
                    'resnet50_v1d_0.37',
                    'mobilenetv2_0.5',
                    #'faster_rcnn_fpn_bn_resnet50_v1b_coco',
                    'resnet50_v1c',
                    'densenet161',
                    'simple_pose_resnet50_v1b',
                    'resnet18_v1b',
                    'darknet53',
                    'fcn_resnet50_ade',
                    'cifar_wideresnet28_10',
                    'simple_pose_resnet101_v1d',
                    'vgg16',
                    'ssd_512_resnet50_v1_coco',
                    'resnet101_v1d_0.73',
                    'squeezenet1.0',
                    'resnet50_v1b',
                    #'faster_rcnn_resnet101_v1d_coco',
                    'ssd_512_mobilenet1.0_voc',
                    'cifar_wideresnet40_8',
                    'cifar_wideresnet16_10',
                    'cifar_resnet110_v2',
                    'resnet101_v1s',
                    'mobilenetv2_0.25',
                    'resnet152_v1c',
                    'se_resnext101_64x4d',
                    #'faster_rcnn_fpn_resnet101_v1d_coco',
                    'resnet50_v1d',
                    'densenet169',
                    'resnet34_v1b',
                    'resnext50_32x4d',
                    'resnet101_v1',
                    'resnet101_v1b',
                    'resnet50_v1s',
                    'mobilenet0.75',
                    'cifar_resnet20_v2',
                    'resnet101_v1d',
                    'vgg11_bn',
                    'resnet18_v2',
                    'vgg11',
                    'simple_pose_resnet101_v1b',
                    'resnext101_32x4d',
                    'resnet50_v2',
                    'vgg16_bn',
                    'mobilenetv2_1.0',
                    'resnet50_v1d_0.48',
                    'resnet50_v1d_0.11',
                    'fcn_resnet101_ade',
                    'simple_pose_resnet152_v1d',
                    'yolo3_mobilenet1.0_coco',
                    'fcn_resnet101_coco']
    # TODO(anisub): add support for other models from gluoncv
    # Not supported today mostly because of broken net.forward calls
    segmentation_models = ['deeplab_resnet50_ade',
                           'psp_resnet101_voc',
                           'deeplab_resnet152_voc',
                           'deeplab_resnet101_ade',
                           'deeplab_resnet152_coco',
                           'psp_resnet101_ade',
                           'deeplab_resnet101_coco',
                           'psp_resnet101_citys',
                           'psp_resnet50_ade',
                           'psp_resnet101_coco',
                           'deeplab_resnet101_voc']
    calib_ssd_models = ["ssd_512_vgg16_atrous_voc",
                        "ssd_300_vgg16_atrous_voc",
                        "ssd_300_vgg16_atrous_coco"]
    calib_inception_models = ["inceptionv3"]
    gluon_models = gluon_models + segmentation_models + \
                   calib_ssd_models + calib_inception_models
    models = gluon_models

    parser = argparse.ArgumentParser(description='Convert a provided FP32 model to a mixed precision model')
    parser.add_argument('--model', type=str, choices=models)
    parser.add_argument('--run-dummy-inference', action='store_true', default=False,
                        help='Will generate random input of shape (1, 3, 224, 224) '
                             'and run a dummy inference forward pass')
    parser.add_argument('--cast-optional-params', action='store_true', default=False,
                        help='If enabled, will try to cast params to target dtype wherever possible')
    args = parser.parse_args()
    logging.basicConfig()
    logger = logging.getLogger('logger')
    logger.setLevel(logging.INFO)

    assert args.model in gluon_models, "Please choose one of the available gluon models: {}".format(gluon_models)
    shape = None
    if args.model in segmentation_models:
        shape = (1, 3, 480, 480)
    elif args.model in calib_ssd_models:
        shape = (1, 3, 512, 544)
    elif args.model in calib_inception_models:
        shape = (1, 3, 299, 299)
    else:
        shape = (1, 3, 224, 224)
    net = gluoncv.model_zoo.get_model(args.model, pretrained=True)
    net.hybridize()
    result_before1 = net.forward(mx.nd.random.uniform(shape=shape))
    net.export("{}".format(args.model))
    net = amp.convert_hybrid_block(net, cast_optional_params=args.cast_optional_params)
    net.export("{}-amp".format(args.model), remove_amp_cast=False)
    if args.run_dummy_inference:
        logger.info("Running inference on the mixed precision model with dummy inputs, batch size: 1")
        result_after = net.forward(mx.nd.random.uniform(shape=shape, dtype=np.float32, ctx=mx.gpu(0)))
        result_after = net.forward(mx.nd.random.uniform(shape=shape, dtype=np.float32, ctx=mx.gpu(0)))
        logger.info("Inference run successfully")
