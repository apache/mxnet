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

import os
import sys
import unittest
try:
    import onnx.backend.test
except ImportError:
    raise ImportError("Onnx and protobuf need to be installed")
CURR_PATH = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
sys.path.insert(0, os.path.join(CURR_PATH, '../'))
import backend as mxnet_backend
import test_cases

# This is a pytest magic variable to load extra plugins
pytest_plugins = "onnx.backend.test.report",

BACKEND_TESTS = onnx.backend.test.BackendTest(mxnet_backend, __name__)

implemented_op_tests = test_cases.IMPLEMENTED_OPERATORS_TEST.get('both',[]) + \
                       test_cases.IMPLEMENTED_OPERATORS_TEST.get('export', [])
for op_tests in implemented_op_tests:
    BACKEND_TESTS.include(op_tests)

basic_models = test_cases.BASIC_MODEL_TESTS.get('both', []) + \
               test_cases.BASIC_MODEL_TESTS.get('export', [])
for basic_model_test in basic_models:
    BACKEND_TESTS.include(basic_model_test)

standard_models = test_cases.STANDARD_MODEL.get('both', []) + \
                  test_cases.STANDARD_MODEL.get('export', [])
for std_model_test in standard_models:
    BACKEND_TESTS.include(std_model_test)

BACKEND_TESTS.exclude('.*bcast.*')


# import all test cases at global scope to make them visible to python.unittest
globals().update(BACKEND_TESTS.enable_report().test_cases)

if __name__ == '__main__':
    unittest.main()
