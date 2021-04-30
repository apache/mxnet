#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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


img_class_models = [
                    'alexnet',
                    'cifar_resnet20_v1',
                    'cifar_resnet56_v1',
                    'cifar_resnet110_v1',
                    'cifar_resnet20_v2',
                    'cifar_resnet56_v2',
                    'cifar_resnet110_v2',
                    'cifar_wideresnet16_10',
                    'cifar_wideresnet28_10',
                    'cifar_wideresnet40_8',
                    'cifar_resnext29_16x64d',
                    'darknet53',
                    'densenet121',
                    'densenet161',
                    'densenet169',
                    'densenet201',
                    'googlenet',
                    'mobilenet1.0',
                    'mobilenet0.75',
                    'mobilenet0.5',
                    'mobilenet0.25',
                    'mobilenetv2_1.0',
                    'mobilenetv2_0.75',
                    'mobilenetv2_0.5',
                    'mobilenetv2_0.25',
                    'mobilenetv3_large',
                    'mobilenetv3_small',
                    'resnest14',
                    'resnest26',
                    'resnest50',
                    'resnest101',
                    'resnest200',
                    'resnest269',
                    'resnet18_v1',
                    'resnet18_v1b_0.89',
                    'resnet18_v2',
                    'resnet34_v1',
                    'resnet34_v2',
                    'resnet50_v1',
                    'resnet50_v1d_0.86',
                    'resnet50_v1d_0.48',
                    'resnet50_v1d_0.37',
                    'resnet50_v1d_0.11',
                    'resnet50_v2',
                    'resnet101_v1',
                    'resnet101_v1d_0.76',
                    'resnet101_v1d_0.73',
                    'resnet101_v2',
                    'resnet152_v1',
                    'resnet152_v2',
                    'resnext50_32x4d',
                    'resnext101_32x4d',
                    'resnext101_64x4d',
                    'senet_154',
                    'se_resnext101_32x4d',
                    'se_resnext101_64x4d',
                    'se_resnext50_32x4d',
                    'squeezenet1.0',
                    'squeezenet1.1',
                    'vgg11',
                    'vgg11_bn',
                    'vgg13',
                    'vgg13_bn',
                    'vgg16',
                    'vgg16_bn',
                    'vgg19',
                    'vgg19_bn',
                    'xception',
                    'inceptionv3'
                   ]

obj_detect_models = [
                    'center_net_resnet18_v1b_voc',
                    'center_net_resnet50_v1b_voc',
                    'center_net_resnet101_v1b_voc',
                    'center_net_resnet18_v1b_coco',
                    'center_net_resnet50_v1b_coco',
                    'center_net_resnet101_v1b_coco',
                    'ssd_300_vgg16_atrous_voc',
                    'ssd_512_vgg16_atrous_voc',
                    'ssd_512_resnet50_v1_voc',
                    'ssd_512_mobilenet1.0_voc',
                    'faster_rcnn_resnet50_v1b_voc',
                    'yolo3_darknet53_voc',
                    'yolo3_mobilenet1.0_voc',
                    'ssd_300_vgg16_atrous_coco',
                    'ssd_512_vgg16_atrous_coco',
                    'ssd_300_resnet34_v1b_coco',
                    'ssd_512_resnet50_v1_coco',
                    'ssd_512_mobilenet1.0_coco',
                    'faster_rcnn_resnet50_v1b_coco',
                    'faster_rcnn_resnet101_v1d_coco',
                    'yolo3_darknet53_coco',
                    'yolo3_mobilenet1.0_coco',
                    'faster_rcnn_fpn_resnet50_v1b_coco',
                    'faster_rcnn_fpn_resnet101_v1d_coco',
                    'mask_rcnn_fpn_resnet18_v1b_coco',
                    'mask_rcnn_resnet18_v1b_coco',
                    'mask_rcnn_resnet50_v1b_coco',
                    'mask_rcnn_resnet101_v1d_coco',
                    'mask_rcnn_fpn_resnet50_v1b_coco',
                    'mask_rcnn_fpn_resnet101_v1d_coco',
                    ]

img_seg_models = [
                    'fcn_resnet50_ade',
                    'fcn_resnet101_ade',
                    'deeplab_resnet50_ade',
                    'deeplab_resnet101_ade',
                    'deeplab_resnest50_ade',
                    'deeplab_resnest101_ade',
                    'deeplab_resnest269_ade',
                    'fcn_resnet101_coco',
                    'deeplab_resnet101_coco',
                    'fcn_resnet101_voc',
                    'deeplab_resnet101_voc',
                    'deeplab_resnet152_voc',
                    'deeplab_resnet50_citys',
                    'deeplab_resnet101_citys',
                    'deeplab_v3b_plus_wideresnet_citys',
                    'danet_resnet50_citys',
                    'danet_resnet101_citys',
                 ]

pose_est_models = [
                    'simple_pose_resnet18_v1b',
                    'simple_pose_resnet50_v1b',
                    'simple_pose_resnet50_v1d',
                    'simple_pose_resnet101_v1b',
                    'simple_pose_resnet101_v1d',
                    'simple_pose_resnet152_v1b',
                    'simple_pose_resnet152_v1d',
                    'alpha_pose_resnet101_v1b_coco',
                    'mobile_pose_resnet18_v1b',
                    'mobile_pose_resnet50_v1b',
                    'mobile_pose_mobilenet1.0',
                    'mobile_pose_mobilenetv2_1.0',
                    'mobile_pose_mobilenetv3_large',
                    'mobile_pose_mobilenetv3_small',
                  ]

act_recg_models = [
                    'inceptionv1_kinetics400',
                    'resnet18_v1b_kinetics400',
                    'resnet34_v1b_kinetics400',
                    'resnet50_v1b_kinetics400',
                    'resnet101_v1b_kinetics400',
                    'resnet152_v1b_kinetics400',
                    'resnet50_v1b_hmdb51',
                    'resnet50_v1b_sthsthv2',
                    'vgg16_ucf101',
                    'inceptionv3_kinetics400',
                    'inceptionv3_ucf101',
                  ]

nlp_models = [  
                'awd_lstm_lm_600',
                'awd_lstm_lm_1150',
                'bert_12_768_12', 
                'bert_24_1024_16',
                'distilbert_6_768_12',
                'ernie_12_768_12',
                'gpt2_117m',
                'gpt2_345m',
                'roberta_12_768_12',
                'roberta_24_1024_16',
                'standard_lstm_lm_200',
                'standard_lstm_lm_650',
                'standard_lstm_lm_1500',
                'transformer_en_de_512',
             ]

type_to_list = {
    'Image Classification' : img_class_models,
    'Object Detection' : obj_detect_models,
    'Image Segmentation' : img_seg_models,
    'Pose Estimation' : pose_est_models,
    'Action Estimation' : act_recg_models,
    'Action Estimation' : act_recg_models,
    'NLP Models' : nlp_models,
}

def generate_model_support_matrix():
    md_string = '*** [GluonCV Pretrained Model Support Matrix](https://cv.gluon.ai/model_zoo/index.html)\n'
    for modle_type in type_to_list:
        if modle_type == 'NLP Models':
            md_string += '### [GluonNLP Pretrained Model Support Matrix](https://nlp.gluon.ai/model_zoo/catalog.html)\n'
        model_list = type_to_list[modle_type]
        md_string += '|%s|\n|:-|\n' % modle_type
        for model in model_list:
            md_string += '|%s|\n' % model
        md_string += '\n'
    try:
        file_name = './model_support_matrix.md'
        with open(file_name, 'w') as f:
            f.write(md_string)
    except Exception as e:
        print('Error writing to file')

generate_model_support_matrix()