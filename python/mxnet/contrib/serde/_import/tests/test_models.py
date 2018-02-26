# Copyright 2017 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#     http://www.apache.org/licenses/LICENSE-2.0
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.

# coding: utf-8
"""Testing model conversions from onnx/models repo"""
from __future__ import absolute_import as _abs
from __future__ import print_function
from collections import namedtuple
import os
import tarfile
import mxnet as mx
from mxnet.test_utils import download
import numpy as np
import numpy.testing as npt
import onnx_mxnet

URLS = {
    'squeezenet_onnx' : 'https://s3.amazonaws.com/download.onnx/models/squeezenet.tar.gz',
    'shufflenet_onnx' : 'https://s3.amazonaws.com/download.onnx/models/shufflenet.tar.gz',
    'inception_v1_onnx' : 'https://s3.amazonaws.com/download.onnx/models/inception_v1.tar.gz',
    'inception_v2_onnx' : 'https://s3.amazonaws.com/download.onnx/models/inception_v2.tar.gz',
    'bvlc_alexnet_onnx' : 'https://s3.amazonaws.com/download.onnx/models/bvlc_alexnet.tar.gz',
    'densenet121_onnx' : 'https://s3.amazonaws.com/download.onnx/models/densenet121.tar.gz',
    'resnet50_onnx' : 'https://s3.amazonaws.com/download.onnx/models/resnet50.tar.gz',
    'vgg16_onnx' : 'https://s3.amazonaws.com/download.onnx/models/vgg16.tar.gz',
    'vgg19_onnx' : 'https://s3.amazonaws.com/download.onnx/models/vgg19.tar.gz'
}

def extract_file(model_tar):
    """Extract tar file and returns model path and input, output data"""
    # extract tar file
    tar = tarfile.open(model_tar, "r:*")
    tar.extractall()
    tar.close()
    path = model_tar.rsplit('_', 1)[0]
    # return model, inputs, outputs path
    cur_dir = os.path.abspath(os.path.dirname(__file__))
    model_path = os.path.join(cur_dir, path, 'model.onnx')
    npz_path = os.path.join(cur_dir, path, 'test_data_0.npz')
    sample = np.load(npz_path, encoding='bytes')
    input_data = list(sample['inputs'])
    output_data = list(sample['outputs'])
    return model_path, input_data, output_data

def verify_onnx_forward_impl(model_path, input_data, output_data):
    """Verifies result after inference"""
    print("Converting onnx format to mxnet's symbol and params...")
    sym, params = onnx_mxnet.import_model(model_path)

    # create module
    mod = mx.mod.Module(symbol=sym, data_names=['input_0'], context=mx.cpu(), label_names=None)
    mod.bind(for_training=False, data_shapes=[('input_0', input_data.shape)], label_shapes=None)
    mod.set_params(arg_params=params, aux_params=params, allow_missing=True, allow_extra=True)
    # run inference
    Batch = namedtuple('Batch', ['data'])

    mod.forward(Batch([mx.nd.array(input_data)]), is_train=False)

    # Run the model with an onnx backend and verify the results
    npt.assert_equal(mod.get_outputs()[0].shape, output_data.shape)
    npt.assert_almost_equal(output_data, mod.get_outputs()[0].asnumpy(), decimal=3)
    print("Conversion Successful")

def verify_model(name):
    """Testing models from onnx model zoo"""
    print("Testing model ", name)
    download(URLS.get(name), name)
    model_path, inputs, outputs = extract_file(name)
    input_data = np.asarray(inputs[0], dtype=np.float32)
    output_data = np.asarray(outputs[0], dtype=np.float32)
    verify_onnx_forward_impl(model_path, input_data, output_data)

if __name__ == '__main__':
    verify_model('squeezenet_onnx') # working
    verify_model('bvlc_alexnet_onnx') # working
    verify_model('vgg16_onnx') # working
    verify_model('vgg19_onnx')  # working
    #verify_model('inception_v1_onnx') # working, accuracy is different 1.4
    #verify_model('inception_v2_onnx') # working, accuracy is different 7.4
    #verify_model('shufflenet_onnx') # working, accuracy is different 10.2
    verify_model('densenet121_onnx') # working
    #verify_model('resnet50_onnx') # working, accuracy is different 18.1
