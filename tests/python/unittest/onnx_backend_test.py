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
# pylint: disable=invalid-name,import-error,wrong-import-position
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import unittest
try:
    import onnx.backend.test
except ImportError:
    raise ImportError("Onnx and protobuf need to be installed")

from os import sys
sys.path.append('../onnx_test_utils')
import backend as mxnet_backend

# This is a pytest magic variable to load extra plugins
pytest_plugins = "onnx.backend.test.report",

backend_test = onnx.backend.test.BackendTest(mxnet_backend, __name__)

implemented_operators = [
    #Generator Functions
    #'test_constant*', # Identity Function
    #'test_random_uniform',
    #'test_random_normal'
    #Arithmetic Operators
    'test_add*',
    'test_sub_bcast_cpu',
    'test_sub_cpu',
    'test_sub_example_cpu',
    'test_mul*',
    'test_div*',
    'test_neg*',
    'test_abs*',
    'test_argmax*',
    'test_argmin*',
    #Hyperbolic functions
    'test_tanh*',
    #Rounding
    'test_ceil*',
    ## Joining and spliting
    # 'test_concat*',  ---Failing test
    #Basic neural network functions
    'test_sigmoid*',
    'test_constant_pad',
    'test_edge_pad',
    'test_reflect_pad',
    'test_relu',
    'test_matmul*',
    #Changing shape and type.
    'test_reshape_*',
    'test_AvgPool2D*'
    #Powers
    'test_reciprocal*',
    'test_sqrt*',
    'test_pow_example',
    #'test_pow',
    #'test_pow_bcast'
    #'test_pow_bcast_axis0'
    ]

for op_test in implemented_operators:
    backend_test.include(op_test)

# import all test cases at global scope to make them visible to python.unittest
globals().update(backend_test
                 .enable_report()
                 .test_cases)

if __name__ == '__main__':
    unittest.main()
