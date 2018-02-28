# Copyright 2017 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#     http://www.apache.org/licenses/LICENSE-2.0
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.
"""onnx test backend wrapper"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import unittest

import onnx.backend.test
from onnx_mxnet import backend as mxnet_backend

# This is a pytest magic variable to load extra plugins
pytest_plugins = 'onnx.backend.test.report'

backend_test = onnx.backend.test.BackendTest(mxnet_backend, __name__)

# will add model tests later
backend_test.exclude('test_bvlc_alexnet')
backend_test.exclude('test_densenet121')
backend_test.exclude('test_inception_v1')
backend_test.exclude('test_inception_v2')
backend_test.exclude('test_resnet50')
backend_test.exclude('test_shufflenet')

# Not implemented
unimplemented_operators = [
    'test_and*',
    'test_clip*',
    'test_equal*',
    'test_gather*',
    'test_greater*',
    'test_hardmax*',
    'test_hardsigmoid*',
    'test_less*',
    'test_logsoftmax*',
    'test_mean*',
    'test_not*',
    'test_or*',
    'test_selu*',
    'test_shape*',
    'test_size*',
    'test_softplus*',
    'test_softsign*',
    'test_thresholdedrelu*',
    'test_top*',
    'test_unsqueeze*',
    'test_xor*',
    'test_Embedding*',
    'test_PReLU*',
    'test_Softplus*',
    #'test_Upsample*',
    'test_operator*',
    'test_constant_cpu'
    ]
for op_test in unimplemented_operators:
    backend_test.exclude(op_test)

# import all test cases at global scope to make them visible to python.unittest
globals().update(backend_test
                 .enable_report()
                 .test_cases)

if __name__ == '__main__':
    unittest.main()
