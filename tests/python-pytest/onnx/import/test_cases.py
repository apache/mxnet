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

"""Test Cases to be run for the import module"""

IMPLEMENTED_OPERATORS_TEST = [
    'test_random_uniform',
    'test_random_normal',
    'test_add',
    'test_sub',
    'test_mul',
    'test_div',
    'test_neg',
    'test_abs',
    'test_sum',
    'test_tanh',
    'test_ceil',
    'test_floor',
    'test_concat',
    'test_sigmoid',
    'test_relu',
    'test_constant_pad',
    'test_edge_pad',
    'test_reflect_pad',
    'test_reduce_min',
    'test_reduce_max',
    'test_reduce_mean',
    'test_reduce_prod',
    'test_squeeze',
    'test_unsqueeze',
    'test_softmax_example',
    'test_softmax_large_number',
    'test_softmax_axis_2',
    'test_transpose',
    'test_globalmaxpool',
    'test_globalaveragepool',
    'test_slice_cpu',
    'test_slice_neg',
    'test_squeeze_',
    'test_reciprocal',
    'test_sqrt',
    'test_pow',
    'test_exp',
    'test_argmax',
    'test_argmin',
    'test_min',
    #pytorch operator tests
    'test_operator_exp',
    'test_operator_maxpool',
    'test_operator_params',
    'test_operator_permute2'
    ]

BASIC_MODEL_TESTS = [
    'test_AvgPool2D',
    'test_BatchNorm',
    'test_ConstantPad2d'
    'test_Conv2d',
    'test_ELU',
    'test_LeakyReLU',
    'test_MaxPool',
    'test_PReLU',
    'test_ReLU',
    'test_Sigmoid',
    'test_Softmax',
    'test_softmax_functional',
    'test_softmax_lastdim',
    'test_Tanh'
    ]

STANDARD_MODEL = [
    'test_bvlc_alexnet',
    'test_densenet121',
    #'test_inception_v1',
    #'test_inception_v2',
    'test_resnet50',
    #'test_shufflenet',
    'test_squeezenet',
    'test_zfnet512',
    'test_vgg19'
    ]
