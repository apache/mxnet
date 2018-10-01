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

"""ONNX test backend wrapper"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import unittest
try:
    import onnx.backend.test
except ImportError:
    raise ImportError("Onnx and protobuf need to be installed")

import backend as mxnet_backend

# This is a pytest magic variable to load extra plugins
pytest_plugins = "onnx.backend.test.report",

BACKEND_TESTS = onnx.backend.test.BackendTest(mxnet_backend, __name__)

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
    'test_cos',
    'test_sin',
    'test_tan',
    'test_acos',
    'test_asin',
    'test_atan'
    'test_ceil',
    'test_floor',
    'test_concat',
    'test_identity',
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
    'test_softmax_example',
    'test_softmax_large_number',
    'test_softmax_axis_2',
    'test_transpose',
    'test_globalmaxpool',
    'test_globalaveragepool',
    # enabling partial test cases for matmul
    'test_matmul_3d',
    'test_matmul_4d',
    'test_slice_cpu',
    'test_slice_neg',
    'test_squeeze_',
    'test_reciprocal',
    'test_sqrt',
    'test_pow',
    'test_exp_',
    'test_argmax',
    'test_argmin',
    'test_min',
    'test_max'
    #pytorch operator tests
    'test_operator_exp',
    'test_operator_maxpool',
    'test_operator_params',
    'test_operator_permute2',
    'test_clip'
    'test_cast',
    'test_depthtospace'
    ]

BASIC_MODEL_TESTS = [
    'test_AvgPool2D',
    'test_BatchNorm',
    'test_ConstantPad2d',
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
    # 'test_inception_v1',
    # 'test_inception_v2',
    'test_resnet50',
    # 'test_shufflenet',
    'test_squeezenet',
    'test_vgg16',
    'test_vgg19'
    ]

for op_test in IMPLEMENTED_OPERATORS_TEST:
    BACKEND_TESTS.include(op_test)

for basic_model_test in BASIC_MODEL_TESTS:
    BACKEND_TESTS.include(basic_model_test)

for std_model_test in STANDARD_MODEL:
    BACKEND_TESTS.include(std_model_test)

BACKEND_TESTS.exclude('.*broadcast.*')
BACKEND_TESTS.exclude('.*bcast.*')


# import all test cases at global scope to make them visible to python.unittest
globals().update(BACKEND_TESTS.enable_report().test_cases)

if __name__ == '__main__':
    unittest.main()
