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
import logging
import tarfile
from collections import namedtuple
import numpy as np
import numpy.testing as npt
from onnx import numpy_helper
from onnx import TensorProto
from mxnet.test_utils import download
from mxnet.contrib import onnx as onnx_mxnet
import mxnet as mx
CURR_PATH = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
sys.path.insert(0, os.path.join(CURR_PATH, '../../python/unittest'))
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
URLS = {
    'bvlc_googlenet':
        'https://s3.amazonaws.com/onnx-mxnet/model-zoo/bvlc_googlenet.tar.gz',
    'bvlc_reference_caffenet':
        'https://s3.amazonaws.com/onnx-mxnet/model-zoo/bvlc_reference_caffenet.tar.gz',
    'bvlc_reference_rcnn_ilsvrc13':
        'https://s3.amazonaws.com/onnx-mxnet/model-zoo/bvlc_reference_rcnn_ilsvrc13.tar.gz',
    'inception_v1':
        'https://s3.amazonaws.com/onnx-mxnet/model-zoo/inception_v1.tar.gz',
    'inception_v2':
        'https://s3.amazonaws.com/onnx-mxnet/model-zoo/inception_v2.tar.gz'
}

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


def forward_pass(sym, arg, aux, data_names, input_data):
    """ Perform forward pass on given data"""
    # create module
    mod = mx.mod.Module(symbol=sym, data_names=data_names, context=mx.cpu(), label_names=None)
    mod.bind(for_training=False, data_shapes=[(data_names[0], input_data.shape)], label_shapes=None)
    mod.set_params(arg_params=arg, aux_params=aux,
                   allow_missing=True, allow_extra=True)
    # run inference
    batch = namedtuple('Batch', ['data'])
    mod.forward(batch([mx.nd.array(input_data)]), is_train=False)

    return mod.get_outputs()[0].asnumpy()


def test_models(model_name, input_shape, output_shape):
    """ Tests Googlenet model for both onnx import and export"""
    model_path, inputs, outputs = get_test_files(model_name)
    logging.info("Translating model from ONNX model zoo to Mxnet")
    sym, arg_params, aux_params = onnx_mxnet.import_model(model_path)
    params = {}
    params.update(arg_params)
    params.update(aux_params)

    dir_path = os.path.dirname(model_path)
    new_model_name = "exported_" + model_name + ".onnx"
    onnx_file = os.path.join(dir_path, new_model_name)

    logging.info("Translating converted model from mxnet to ONNX")
    converted_model_path = onnx_mxnet.export_model(sym, params, [input_shape], np.float32, onnx_file)

    sym, arg_params, aux_params = onnx_mxnet.import_model(converted_model_path)

    metadata = onnx_mxnet.get_model_metadata(converted_model_path)
    assert len(metadata) == 2
    assert metadata.get('input_tensor_data')
    assert metadata.get('input_tensor_data')[0][1] == input_shape
    assert metadata.get('output_tensor_data')
    assert metadata.get('output_tensor_data')[0][1] == output_shape
    data_names = [input_name[0] for input_name in metadata.get('input_tensor_data')]

    logging.info("Running inference on onnx re-import model in mxnet")
    # run test for each test file
    for input_data, output_data in zip(inputs, outputs):
        result = forward_pass(sym, arg_params, aux_params, data_names, input_data)

        # verify the results
        npt.assert_equal(result.shape, output_data.shape)
        npt.assert_almost_equal(output_data, result, decimal=3)
    logging.info(model_name + " conversion successful")


def test_model_accuracy(model_name, input_shape):
    """ Imports ONNX model, runs inference, exports and imports back
        run inference, compare result with the previous inference result"""
    model_path, inputs, outputs = get_test_files(model_name)
    logging.info("Translating model from ONNX model zoo to Mxnet")
    sym, arg_params, aux_params = onnx_mxnet.import_model(model_path)

    metadata = onnx_mxnet.get_model_metadata(model_path)
    data_names = [input_name[0] for input_name in metadata.get('input_tensor_data')]

    expected_result= []
    for input_data, output_data in zip(inputs, outputs):
        result = forward_pass(sym, arg_params, aux_params, data_names, input_data)
        expected_result.append(result)

    params = {}
    params.update(arg_params)
    params.update(aux_params)

    dir_path = os.path.dirname(model_path)
    new_model_name = "exported_" + model_name + ".onnx"
    onnx_file = os.path.join(dir_path, new_model_name)

    logging.info("Translating converted model from mxnet to ONNX")
    converted_model_path = onnx_mxnet.export_model(sym, params, [input_shape], np.float32,
                                                   onnx_file)

    sym, arg_params, aux_params = onnx_mxnet.import_model(converted_model_path)

    metadata = onnx_mxnet.get_model_metadata(converted_model_path)
    data_names = [input_name[0] for input_name in metadata.get('input_tensor_data')]

    actual_result = []
    for input_data, output_data in zip(inputs, outputs):
        result = forward_pass(sym, arg_params, aux_params, data_names, input_data)
        actual_result.append(result)

    # verify the results
    for expected, actual in zip(expected_result, actual_result):
        npt.assert_equal(expected.shape, actual.shape)
        npt.assert_almost_equal(expected, actual, decimal=3)


if __name__ == '__main__':
    test_models("bvlc_googlenet", (1, 3, 224, 224), (1, 1000))
    test_models("bvlc_reference_caffenet", (1, 3, 224, 224), (1, 1000))
    test_models("bvlc_reference_rcnn_ilsvrc13", (1, 3, 224, 224), (1, 200))

    # Comparing MXNet inference result, since MXNet results don't match
    # ONNX expected results due to AveragePool issue github issue(#10194)
    test_model_accuracy("inception_v1", (1, 3, 224, 224))
    test_model_accuracy("inception_v2", (1, 3, 224, 224))
