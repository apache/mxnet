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

"""Testing super_resolution model conversion"""
from __future__ import absolute_import as _abs
from __future__ import print_function
from collections import namedtuple
import logging
import numpy as np
from PIL import Image
import mxnet as mx
from mxnet.test_utils import download
import mxnet.contrib.onnx as onnx_mxnet

# set up logger
logging.basicConfig()
LOGGER = logging.getLogger()
LOGGER.setLevel(logging.INFO)

def download_onnx_model():
    """Download the onnx model"""
    model_url = 'https://s3.amazonaws.com/onnx-mxnet/examples/super_resolution.onnx'
    download(model_url, 'super_resolution.onnx')

def import_onnx():
    """Import the onnx model into mxnet"""
    LOGGER.info("Converting onnx format to mxnet's symbol and params...")
    sym, params = onnx_mxnet.import_model('super_resolution.onnx')
    assert sym is not None
    assert params is not None

    inputs = sym.list_inputs()
    assert len(inputs) == 9
    for i, input_param in enumerate(['param_7', 'param_5', 'param_3', 'param_1',
                                     'input_0', 'param_0', 'param_2', 'param_4', 'param_6']):
        assert inputs[i] == input_param

    assert len(sym.list_outputs()) == 1
    assert sym.list_outputs()[0] == 'reshape5_output'

    assert len(sym.list_attr()) == 1
    assert sym.list_attr()['shape'] == '(1L, 1L, 672L, 672L)'

    attrs_keys = sym.attr_dict().keys()
    assert len(attrs_keys) == 19
    for i, key_item in enumerate(['reshape4', 'param_5', 'param_4', 'param_7',
                                  'param_6', 'param_1', 'param_0', 'param_3',
                                  'param_2', 'reshape2', 'reshape3', 'reshape0',
                                  'reshape1', 'convolution2', 'convolution3',
                                  'convolution0', 'convolution1', 'reshape5',
                                  'transpose0']):
        assert attrs_keys[i] == key_item

    param_keys = params.keys()
    assert len(param_keys) == 8
    for i, param_item in enumerate(['param_5', 'param_4', 'param_7', 'param_6',
                                    'param_1', 'param_0', 'param_3', 'param_2']):
        assert param_keys[i] == param_item
    return sym, params

def get_test_image():
    """Download and process the test image"""
    # Load test image
    input_image_dim = 224
    img_url = 'https://s3.amazonaws.com/onnx-mxnet/examples/super_res_input.jpg'
    download(img_url, 'super_res_input.jpg')
    img = Image.open('super_res_input.jpg').resize((input_image_dim, input_image_dim))
    img_ycbcr = img.convert("YCbCr")
    img_y, img_cb, img_cr = img_ycbcr.split()
    input_image = np.array(img_y)[np.newaxis, np.newaxis, :, :]
    return input_image, img_cb, img_cr

def perform_inference((sym, params), (input_img, img_cb, img_cr)):
    """Perform inference on image using mxnet"""
    # create module
    mod = mx.mod.Module(symbol=sym, data_names=['input_0'], label_names=None)
    mod.bind(for_training=False, data_shapes=[('input_0', input_img.shape)])
    mod.set_params(arg_params=params, aux_params=None)

    # run inference
    batch = namedtuple('Batch', ['data'])
    mod.forward(batch([mx.nd.array(input_img)]))

    # Save the result
    img_out_y = Image.fromarray(np.uint8(mod.get_outputs()[0][0][0].
                                         asnumpy().clip(0, 255)), mode='L')

    result_img = Image.merge(
        "YCbCr", [img_out_y,
                  img_cb.resize(img_out_y.size, Image.BICUBIC),
                  img_cr.resize(img_out_y.size, Image.BICUBIC)]).convert("RGB")
    output_img_dim = 672
    assert result_img.size == (output_img_dim, output_img_dim)
    result_img.save("super_res_output.jpg")

if __name__ == '__main__':
    download_onnx_model()
    perform_inference(import_onnx(), get_test_image())
