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

def import_onnx():
    """Import the onnx model into mxnet"""
    model_url = 'https://s3.amazonaws.com/onnx-mxnet/examples/super_resolution.onnx'
    download(model_url, 'super_resolution.onnx')

    LOGGER.info("Converting onnx format to mxnet's symbol and params...")
    sym, arg_params, aux_params = onnx_mxnet.import_model('super_resolution.onnx')
    LOGGER.info("Successfully Converted onnx format to mxnet's symbol and params...")
    return sym, arg_params, aux_params

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

def perform_inference(sym, arg_params, aux_params, input_img, img_cb, img_cr):
    """Perform inference on image using mxnet"""
    # To fetch the data names of the input to the model we list the inputs of the symbol graph
    # and exclude the argument and auxiliary parameters from the list
    data_names = [graph_input for graph_input in sym.list_inputs()
                  if graph_input not in arg_params and graph_input not in aux_params]
    # create module
    mod = mx.mod.Module(symbol=sym, data_names=data_names, label_names=None)
    mod.bind(for_training=False, data_shapes=[(data_names[0], input_img.shape)])
    mod.set_params(arg_params=arg_params, aux_params=aux_params)

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
    LOGGER.info("Super Resolution example success.")
    result_img.save("super_res_output.jpg")
    return result_img

if __name__ == '__main__':
    MX_SYM, MX_ARG_PARAM, MX_AUX_PARAM = import_onnx()
    INPUT_IMG, IMG_CB, IMG_CR = get_test_image()
    perform_inference(MX_SYM, MX_ARG_PARAM, MX_AUX_PARAM, INPUT_IMG, IMG_CB, IMG_CR)
