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

"""Tests for individual operators"""
# pylint: disable=import-error,no-self-use

from __future__ import absolute_import
import unittest
import numpy as np
import numpy.testing as npt
from onnx import helper
import backend as mxnet_backend
from common import with_seed

class TestLayers(unittest.TestCase):
    """Tests for different layers comparing output with numpy operators.
    Temporary file until we have a corresponding test in onnx-backend_test
    for these operators."""

    @with_seed(0)
    def test_reduce_max(self):
        """Test for ReduceMax operator"""
        node_def = helper.make_node("ReduceMax", ["input1"], ["output"], axes=[1, 0], keepdims=1)
        input1 = np.random.ranf([3, 10]).astype("float32")
        output = mxnet_backend.run_node(node_def, [input1])[0]
        numpy_op = np.max(input1, axis=(1, 0), keepdims=True)
        npt.assert_almost_equal(output, numpy_op)

    @with_seed(0)
    def test_reduce_mean(self):
        """Test for ReduceMean operator"""
        node_def = helper.make_node("ReduceMean", ["input1"], ["output"], axes=[1, 0], keepdims=1)
        input1 = np.random.ranf([3, 10]).astype("float32")
        output = mxnet_backend.run_node(node_def, [input1])[0]
        numpy_op = np.mean(input1, axis=(1, 0), keepdims=True)
        npt.assert_almost_equal(output, numpy_op, decimal=5)

    @with_seed(0)
    def test_reduce_min(self):
        """Test for ReduceMin operator"""
        node_def = helper.make_node("ReduceMin", ["input1"], ["output"], axes=[1, 0], keepdims=1)
        input1 = np.random.ranf([3, 10]).astype("float32")
        output = mxnet_backend.run_node(node_def, [input1])[0]
        numpy_op = np.min(input1, axis=(1, 0), keepdims=True)
        npt.assert_almost_equal(output, numpy_op)

    @with_seed(0)
    def test_reduce_sum(self):
        """Test for ReduceSum operator"""
        node_def = helper.make_node("ReduceSum", ["input1"], ["output"], axes=[1, 0], keepdims=1)
        input1 = np.random.ranf([3, 10]).astype("float32")
        output = mxnet_backend.run_node(node_def, [input1])[0]
        numpy_op = np.sum(input1, axis=(1, 0), keepdims=True)
        npt.assert_almost_equal(output, numpy_op, decimal=5)

    @with_seed(0)
    def test_reduce_prod(self):
        """Test for ReduceProd operator"""
        node_def = helper.make_node("ReduceProd", ["input1"], ["output"], axes=[1, 0], keepdims=1)
        input1 = np.random.ranf([3, 10]).astype("float32")
        output = mxnet_backend.run_node(node_def, [input1])[0]
        numpy_op = np.prod(input1, axis=(1, 0), keepdims=True)
        npt.assert_almost_equal(output, numpy_op, decimal=5)

    @with_seed(0)  
    def test_squeeze(self):
        """Test for Squeeze operator"""
        node_def = helper.make_node("Squeeze", ["input1"], ["output"], axes=[1, 3])
        input1 = np.random.ranf([3, 1, 2, 1, 4]).astype("float32")
        output = mxnet_backend.run_node(node_def, [input1])[0]
        npt.assert_almost_equal(output, np.squeeze(input1, axis=[1, 3]))

if __name__ == '__main__':
    unittest.main()
