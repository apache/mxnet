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

"""
Tests for individual operators
This module contains operator tests which currently do not exist on
ONNX backend test framework. Once we have PRs on the ONNX repo and get
those PRs merged, this file will get EOL'ed.
"""
# pylint: disable=too-many-locals,wrong-import-position,import-error
from __future__ import absolute_import
import sys
import os
import unittest
import logging
import hashlib
import tarfile
from collections import namedtuple
import numpy as np
import numpy.testing as npt
from onnx import helper
from onnx import numpy_helper
from onnx import TensorProto
from mxnet.test_utils import download
from mxnet.contrib import onnx as onnx_mxnet
import mxnet as mx
CURR_PATH = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
sys.path.insert(0, os.path.join(CURR_PATH, '../../python/unittest'))
from common import with_seed
import backend as mxnet_backend


URLS = {
    'bvlc_googlenet' :
        'https://s3.amazonaws.com/onnx-mxnet/model-zoo/bvlc_googlenet.tar.gz',
    'bvlc_reference_caffenet' :
        'https://s3.amazonaws.com/onnx-mxnet/model-zoo/bvlc_reference_caffenet.tar.gz',
    'bvlc_reference_rcnn_ilsvrc13' :
        'https://s3.amazonaws.com/onnx-mxnet/model-zoo/bvlc_reference_rcnn_ilsvrc13.tar.gz',
}

@with_seed()
def test_reduce_max():
    """Test for ReduceMax operator"""
    node_def = helper.make_node("ReduceMax", ["input1"], ["output"], axes=[1, 0], keepdims=1)
    input1 = np.random.ranf([3, 10]).astype("float32")
    output = mxnet_backend.run_node(node_def, [input1])[0]
    numpy_op = np.max(input1, axis=(1, 0), keepdims=True)
    npt.assert_almost_equal(output, numpy_op)

@with_seed()
def test_reduce_mean():
    """Test for ReduceMean operator"""
    node_def = helper.make_node("ReduceMean", ["input1"], ["output"], axes=[1, 0], keepdims=1)
    input1 = np.random.ranf([3, 10]).astype("float32")
    output = mxnet_backend.run_node(node_def, [input1])[0]
    numpy_op = np.mean(input1, axis=(1, 0), keepdims=True)
    npt.assert_almost_equal(output, numpy_op, decimal=5)

@with_seed()
def test_reduce_min():
    """Test for ReduceMin operator"""
    node_def = helper.make_node("ReduceMin", ["input1"], ["output"], axes=[1, 0], keepdims=1)
    input1 = np.random.ranf([3, 10]).astype("float32")
    output = mxnet_backend.run_node(node_def, [input1])[0]
    numpy_op = np.min(input1, axis=(1, 0), keepdims=True)
    npt.assert_almost_equal(output, numpy_op)

@with_seed()
def test_reduce_sum():
    """Test for ReduceSum operator"""
    node_def = helper.make_node("ReduceSum", ["input1"], ["output"], axes=[1, 0], keepdims=1)
    input1 = np.random.ranf([3, 10]).astype("float32")
    output = mxnet_backend.run_node(node_def, [input1])[0]
    numpy_op = np.sum(input1, axis=(1, 0), keepdims=True)
    npt.assert_almost_equal(output, numpy_op, decimal=5)

@with_seed()
def test_reduce_prod():
    """Test for ReduceProd operator"""
    node_def = helper.make_node("ReduceProd", ["input1"], ["output"], axes=[1, 0], keepdims=1)
    input1 = np.random.ranf([3, 10]).astype("float32")
    output = mxnet_backend.run_node(node_def, [input1])[0]
    numpy_op = np.prod(input1, axis=(1, 0), keepdims=True)
    npt.assert_almost_equal(output, numpy_op, decimal=5)

@with_seed()
def test_squeeze():
    """Test for Squeeze operator"""
    node_def = helper.make_node("Squeeze", ["input1"], ["output"], axes=[1, 3])
    input1 = np.random.ranf([3, 1, 2, 1, 4]).astype("float32")
    output = mxnet_backend.run_node(node_def, [input1])[0]
    npt.assert_almost_equal(output, np.squeeze(input1, axis=[1, 3]))

def test_super_resolution_example():
    """Test the super resolution example in the example/onnx folder"""
    sys.path.insert(0, os.path.join(CURR_PATH, '../../../example/onnx/'))
    import super_resolution

    sym, arg_params, aux_params = super_resolution.import_onnx()
    assert sym is not None
    assert arg_params is not None

    inputs = sym.list_inputs()
    assert len(inputs) == 9
    for i, input_param in enumerate(['9', '7', '5', '3', '1', '2', '4', '6', '8']):
        assert inputs[i] == input_param

    assert len(sym.list_outputs()) == 1
    assert sym.list_outputs()[0] == 'reshape5_output'

    attrs_keys = sym.attr_dict().keys()
    assert len(attrs_keys) == 23
    for i, key_item in enumerate(['reshape4', 'convolution2', 'convolution0',
                                  'transpose0', '6', 'reshape0', 'reshape2',
                                  'reshape3', '3', 'reshape1', '5', '4', '7',
                                  'convolution1', '9', '2', 'convolution3',
                                  'reshape5', '8', 'pad1', 'pad0', 'pad3',
                                  'pad2']):
        assert key_item in attrs_keys

    param_keys = arg_params.keys()
    assert len(param_keys) == 8
    for i, param_item in enumerate(['3', '2', '5', '4', '7', '6', '9', '8']):
        assert param_item in param_keys

    logging.info("Asserted the result of the onnx model conversion")

    output_img_dim = 672
    input_image, img_cb, img_cr = super_resolution.get_test_image()
    result_img = super_resolution.perform_inference(sym, arg_params, aux_params,
                                                    input_image, img_cb, img_cr)

    assert hashlib.md5(result_img.tobytes()).hexdigest() == '0d98393a49b1d9942106a2ed89d1e854'
    assert result_img.size == (output_img_dim, output_img_dim)

def get_test_files(name):
    """Extract tar file and returns model path and input, output data"""
    tar_name = download(URLS.get(name), dirname=CURR_PATH.__str__())
    # extract tar file
    tar_path = os.path.join(CURR_PATH, tar_name)
    tar = tarfile.open(tar_path.__str__(), "r:*")
    tar.extractall(path=CURR_PATH.__str__())
    tar.close()
    data_dir = os.path.join(CURR_PATH, name)
    model_path = os.path.join(data_dir, 'model.onnx')

    inputs = []
    outputs = []
    # get test files
    for test_file in os.listdir(data_dir):
        case_dir = os.path.join(data_dir, test_file)
        # skip the non-dir files
        if not os.path.isdir(case_dir):
            continue
        input_file = os.path.join(case_dir, 'input_0.pb')
        input_tensor = TensorProto()
        with open(input_file, 'rb') as proto_file:
            input_tensor.ParseFromString(proto_file.read())
        inputs.append(numpy_helper.to_array(input_tensor))

        output_tensor = TensorProto()
        output_file = os.path.join(case_dir, 'output_0.pb')
        with open(output_file, 'rb') as proto_file:
            output_tensor.ParseFromString(proto_file.read())
        outputs.append(numpy_helper.to_array(output_tensor))

    return model_path, inputs, outputs

def test_bvlc_googlenet():
    """ Tests Googlenet model"""
    model_path, inputs, outputs = get_test_files('bvlc_googlenet')
    logging.info("Translating Googlenet model from ONNX to Mxnet")
    sym, arg_params, aux_params = onnx_mxnet.import_model(model_path)

    # run test for each test file
    for input_data, output_data in zip(inputs, outputs):
        # create module
        data_names = [graph_input for graph_input in sym.list_inputs()
                      if graph_input not in arg_params and graph_input not in aux_params]
        mod = mx.mod.Module(symbol=sym, data_names=data_names, context=mx.cpu(), label_names=None)
        mod.bind(for_training=False, data_shapes=[(data_names[0], input_data.shape)], label_shapes=None)
        mod.set_params(arg_params=arg_params, aux_params=aux_params,
                       allow_missing=True, allow_extra=True)
        # run inference
        batch = namedtuple('Batch', ['data'])
        mod.forward(batch([mx.nd.array(input_data)]), is_train=False)

        # verify the results
        npt.assert_equal(mod.get_outputs()[0].shape, output_data.shape)
        npt.assert_almost_equal(output_data, mod.get_outputs()[0].asnumpy(), decimal=3)
    logging.info("Googlenet model conversion Successful")

def test_bvlc_reference_caffenet():
    """Tests the bvlc cafenet model"""
    model_path, inputs, outputs = get_test_files('bvlc_reference_caffenet')
    logging.info("Translating Caffenet model from ONNX to Mxnet")
    sym, arg_params, aux_params = onnx_mxnet.import_model(model_path)

    # run test for each test file
    for input_data, output_data in zip(inputs, outputs):
        # create module
        data_names = [graph_input for graph_input in sym.list_inputs()
                      if graph_input not in arg_params and graph_input not in aux_params]
        mod = mx.mod.Module(symbol=sym, data_names=data_names, context=mx.cpu(), label_names=None)
        mod.bind(for_training=False, data_shapes=[(data_names[0], input_data.shape)], label_shapes=None)
        mod.set_params(arg_params=arg_params, aux_params=aux_params,
                       allow_missing=True, allow_extra=True)
        # run inference
        batch = namedtuple('Batch', ['data'])
        mod.forward(batch([mx.nd.array(input_data)]), is_train=False)

        # verify the results
        npt.assert_equal(mod.get_outputs()[0].shape, output_data.shape)
        npt.assert_almost_equal(output_data, mod.get_outputs()[0].asnumpy(), decimal=3)
    logging.info("Caffenet model conversion Successful")

def test_bvlc_rcnn_ilsvrc13():
    """Tests the bvlc rcnn model"""
    model_path, inputs, outputs = get_test_files('bvlc_reference_rcnn_ilsvrc13')
    logging.info("Translating rcnn_ilsvrc13 model from ONNX to Mxnet")
    sym, arg_params, aux_params = onnx_mxnet.import_model(model_path)

    # run test for each test file
    for input_data, output_data in zip(inputs, outputs):
        # create module
        data_names = [graph_input for graph_input in sym.list_inputs()
                      if graph_input not in arg_params and graph_input not in aux_params]
        mod = mx.mod.Module(symbol=sym, data_names=data_names, context=mx.cpu(), label_names=None)
        mod.bind(for_training=False, data_shapes=[(data_names[0], input_data.shape)], label_shapes=None)
        mod.set_params(arg_params=arg_params, aux_params=aux_params,
                       allow_missing=True, allow_extra=True)
        # run inference
        batch = namedtuple('Batch', ['data'])
        mod.forward(batch([mx.nd.array(input_data)]), is_train=False)

        # verify the results
        npt.assert_equal(mod.get_outputs()[0].shape, output_data.shape)
        npt.assert_almost_equal(output_data, mod.get_outputs()[0].asnumpy(), decimal=3)
    logging.info("rcnn_ilsvrc13 model conversion Successful")


if __name__ == '__main__':
    unittest.main()
