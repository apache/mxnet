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


# pylint: disable=too-many-locals,wrong-import-position,import-error
from __future__ import absolute_import
import sys
import os
import unittest
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
        'https://s3.amazonaws.com/download.onnx/models/opset_8/bvlc_googlenet.tar.gz',
    'bvlc_reference_caffenet':
        'https://s3.amazonaws.com/download.onnx/models/opset_8/bvlc_reference_caffenet.tar.gz',
    'bvlc_reference_rcnn_ilsvrc13':
        'https://s3.amazonaws.com/download.onnx/models/opset_8/bvlc_reference_rcnn_ilsvrc13.tar.gz',
    'inception_v1':
        'https://s3.amazonaws.com/download.onnx/models/opset_8/inception_v1.tar.gz',
    'inception_v2':
        'https://s3.amazonaws.com/download.onnx/models/opset_8/inception_v2.tar.gz'
}

test_model_path = "https://s3.amazonaws.com/onnx-mxnet/test_model.onnx"

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


class TestModel(unittest.TestCase):
    """ Tests for models.
    Tests are dynamically added.
    Therefore edit test_models to add more tests.
    """
    def test_import_export(self):
        def get_model_results(modelpath):
            symbol, args, aux = onnx_mxnet.import_model(modelpath)

            data = onnx_mxnet.get_model_metadata(modelpath)
            data_names = [input_name[0] for input_name in data.get('input_tensor_data')]

            result = []
            for input_data, output_data in zip(inputs, outputs):
                output = forward_pass(symbol, args, aux, data_names, input_data)
                result.append(output)
            return symbol, args, aux, result, data

        for test in test_cases:
            model_name, input_shape, output_shape = test
            with self.subTest(model_name):
                model_path, inputs, outputs = get_test_files(model_name)
                logging.info("Translating " + model_name + " from ONNX model zoo to MXNet")

                sym, arg_params, aux_params, expected_result, _ = get_model_results(model_path)

                params = {}
                params.update(arg_params)
                params.update(aux_params)

                dir_path = os.path.dirname(model_path)
                new_model_name = "exported_" + model_name + ".onnx"
                onnx_file = os.path.join(dir_path, new_model_name)

                logging.info("Translating converted model from mxnet to ONNX")
                converted_model_path = onnx_mxnet.export_model(sym, params, [input_shape], np.float32, onnx_file)

                sym, arg_params, aux_params, actual_result, metadata = get_model_results(converted_model_path)

                assert len(metadata) == 2
                assert metadata.get('input_tensor_data')
                assert metadata.get('input_tensor_data')[0][1] == input_shape
                assert metadata.get('output_tensor_data')
                assert metadata.get('output_tensor_data')[0][1] == output_shape

                # verify the results
                for expected, actual in zip(expected_result, actual_result):
                    npt.assert_equal(expected.shape, actual.shape)
                    npt.assert_almost_equal(expected, actual, decimal=3)

                logging.info(model_name + " conversion successful")

    def test_nodims_import(self):
        # Download test model without dims mentioned in params
        test_model = download(test_model_path, dirname=CURR_PATH.__str__())
        input_data = np.array([0.2, 0.5])
        nd_data = mx.nd.array(input_data).expand_dims(0)
        sym, arg_params, aux_params = onnx_mxnet.import_model(test_model)
        model_metadata = onnx_mxnet.get_model_metadata(test_model)
        input_names = [inputs[0] for inputs in model_metadata.get('input_tensor_data')]
        output_data = forward_pass(sym, arg_params, aux_params, input_names, nd_data)
        assert(output_data.shape == (1,1))

# test_case = ("model name", input shape, output shape)
test_cases = [
    ("bvlc_googlenet", (1, 3, 224, 224), (1, 1000)),
    ("bvlc_reference_caffenet", (1, 3, 224, 224), (1, 1000)),
    ("bvlc_reference_rcnn_ilsvrc13", (1, 3, 224, 224), (1, 200)),
    ("inception_v1", (1, 3, 224, 224), (1, 1000)),
    ("inception_v2", (1, 3, 224, 224), (1, 1000))
]


if __name__ == '__main__':
    unittest.main()
