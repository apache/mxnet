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
try:
    import onnx.backend.test
except ImportError:
    raise ImportError("Onnx and protobuf need to be installed")

import test_cases


def prepare_tests(backend, operation):
    """
    Prepare the test list
    :param backend: mxnet/gluon backend
    :param operation: str. export or import
    :return: backend test list
    """
    BACKEND_TESTS = onnx.backend.test.BackendTest(backend, __name__)
    implemented_ops = test_cases.IMPLEMENTED_OPERATORS_TEST.get('both', []) + \
                      test_cases.IMPLEMENTED_OPERATORS_TEST.get(operation, [])

    for op_test in implemented_ops:
        BACKEND_TESTS.include(op_test)

    basic_models = test_cases.BASIC_MODEL_TESTS.get('both', []) + \
                   test_cases.BASIC_MODEL_TESTS.get(operation, [])

    for basic_model_test in basic_models:
        BACKEND_TESTS.include(basic_model_test)

    std_models = test_cases.STANDARD_MODEL.get('both', []) + \
                 test_cases.STANDARD_MODEL.get(operation, [])

    for std_model_test in std_models:
        BACKEND_TESTS.include(std_model_test)

    BACKEND_TESTS.exclude('.*bcast.*')

    return BACKEND_TESTS
