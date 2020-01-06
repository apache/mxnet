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
import unittest
import backend as mxnet_backend
import logging

operations = ['import', 'export']
backends = ['mxnet', 'gluon']
# This is a pytest magic variable to load extra plugins
pytest_plugins = "onnx.backend.test.report",


def test_suite(backend_tests):  # type: () -> unittest.TestSuite
    '''
    TestSuite that can be run by TestRunner
    This has been borrowed from onnx/onnx/backend/test/runner/__init__.py,
    since Python3 cannot sort objects of type 'Type' as Runner.test_suite()
    expects.
    '''
    suite = unittest.TestSuite()
    for case in backend_tests.test_cases.values():
        suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(case))
    return suite


def prepare_tests(backend, oper):
    """
    Prepare the test list
    :param backend: mxnet/gluon backend
    :param oper: str. export or import
    :return: backend test list
    """
    BACKEND_TESTS = onnx.backend.test.BackendTest(backend, __name__)
    implemented_ops = test_cases.IMPLEMENTED_OPERATORS_TEST.get('both', []) + \
                      test_cases.IMPLEMENTED_OPERATORS_TEST.get(oper, [])

    for op_test in implemented_ops:
        BACKEND_TESTS.include(op_test)

    basic_models = test_cases.BASIC_MODEL_TESTS.get('both', []) + \
                   test_cases.BASIC_MODEL_TESTS.get(oper, [])

    for basic_model_test in basic_models:
        BACKEND_TESTS.include(basic_model_test)

    std_models = test_cases.STANDARD_MODEL.get('both', []) + \
                 test_cases.STANDARD_MODEL.get(oper, [])

    for std_model_test in std_models:
        BACKEND_TESTS.include(std_model_test)

    # Tests for scalar ops are in test_node.py
    BACKEND_TESTS.exclude('.*scalar.*')

    return BACKEND_TESTS


for bkend in backends:
    for operation in operations:
        log = logging.getLogger(bkend + operation)
        if bkend == 'gluon' and operation == 'export':
            log.warning('Gluon->ONNX export not implemented. Skipping tests...')
            continue
        log.info('Executing tests for ' + bkend + ' backend: ' + operation)
        mxnet_backend.MXNetBackend.set_params(bkend, operation)
        BACKEND_TESTS = prepare_tests(mxnet_backend, operation)
        unittest.TextTestRunner().run(test_suite(BACKEND_TESTS.enable_report()))
