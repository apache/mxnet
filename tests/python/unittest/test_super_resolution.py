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
import mxnet as mx
from mxnet.test_utils import download
import mxnet.contrib.onnx._import as onnx_mxnet
import numpy as np
from PIL import Image

model_url = 'https://s3.amazonaws.com/onnx-mxnet/examples/super_resolution.onnx'

download(model_url, 'super_resolution.onnx')

print("Converting onnx format to mxnet's symbol and params...")
sym, params = onnx_mxnet.import_model('super_resolution.onnx')

# Load test image
input_image_dim = 224
output_image_dim = 672
img_url = 'https://s3.amazonaws.com/onnx-mxnet/examples/super_res_input.jpg'
download(img_url, 'super_res_input.jpg')
img = Image.open('super_res_input.jpg').resize((input_image_dim, input_image_dim))
img_ycbcr = img.convert("YCbCr")
img_y, img_cb, img_cr = img_ycbcr.split()
x = np.array(img_y)[np.newaxis, np.newaxis, :, :]

# create module
mod = mx.mod.Module(symbol=sym, data_names=['input_0'], label_names=None)
mod.bind(for_training=False, data_shapes=[('input_0', x.shape)])
mod.set_params(arg_params=params, aux_params=None)

# run inference
Batch = namedtuple('Batch', ['data'])
mod.forward(Batch([mx.nd.array(x)]))

# Save the result
img_out_y = Image.fromarray(np.uint8(mod.get_outputs()[0][0][0].asnumpy().clip(0, 255)), mode='L')

result_img = Image.merge(
    "YCbCr", [img_out_y,
              img_cb.resize(img_out_y.size, Image.BICUBIC),
              img_cr.resize(img_out_y.size, Image.BICUBIC)]).convert("RGB")
result_img.save("super_res_output.jpg")
