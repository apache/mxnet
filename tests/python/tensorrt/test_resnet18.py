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

from mxnet.gluon.model_zoo import vision
from mxnet.test_utils import assert_almost_equal
import mxnet as mx
import numpy as np
import os

batch_shape = (1, 3, 224, 224)
url = 'https://github.com/dmlc/web-data/blob/master/mxnet/doc/tutorials/python/predict_image/cat.jpg?raw=true'
model_file_name = 'resnet18_v2_trt_test'

def get_image(image_url):
    fname = mx.test_utils.download(image_url, fname=image_url.split('/')[-1].split('?')[0])
    img = mx.image.imread(fname)
    img = mx.image.imresize(img, 224, 224)  # Resize
    img = img.transpose((2, 0, 1))  # Channel first
    img = img.expand_dims(axis=0)  # Batchify
    img = mx.nd.cast(img, dtype=np.float32)
    return img / 255.0

def test_tensorrt_resnet18_feature_vect():
    print("downloading sample input")
    input_data = get_image(url)
    gluon_resnet18 = vision.resnet18_v2(pretrained=True)
    gluon_resnet18.hybridize()
    gluon_resnet18.forward(input_data)
    gluon_resnet18.export(model_file_name)
    sym, arg_params, aux_params = mx.model.load_checkpoint(model_file_name, 0)

    executor = sym.simple_bind(ctx=mx.gpu(), data=batch_shape,
                               grad_req='null', force_rebind=True)
    executor.copy_params_from(arg_params, aux_params)
    y = executor.forward(is_train=False, data=input_data)
    trt_sym = sym.get_backend_symbol('TensorRT')
    mx.contrib.tensorrt.init_tensorrt_params(trt_sym, arg_params, aux_params)
    original_precision_value = mx.contrib.tensorrt.get_use_fp16()
    try:
        mx.contrib.tensorrt.set_use_fp16(True)
        executor = trt_sym.simple_bind(ctx=mx.gpu(), data=batch_shape,
                                       grad_req='null', force_rebind=True)
        executor.copy_params_from(arg_params, aux_params)
        y_trt = executor.forward(is_train=False, data=input_data)
        mx.contrib.tensorrt.set_use_fp16(False)
        executor = trt_sym.simple_bind(ctx=mx.gpu(), data=batch_shape,
                                       grad_req='null', force_rebind=True)
        executor.copy_params_from(arg_params, aux_params)
        y_trt_fp32 = executor.forward(is_train=False, data=input_data)
        no_trt_output = y[0].asnumpy()[0]
        trt_output = y_trt[0].asnumpy()[0]
        trt_fp32_output = y_trt_fp32[0].asnumpy()[0]
        assert_almost_equal(no_trt_output, trt_output, 1e-1, 1e-2)
        assert_almost_equal(no_trt_output, trt_fp32_output, 1e-4, 1e-4)
    finally:
        mx.contrib.tensorrt.set_use_fp16(original_precision_value)

if __name__ == '__main__':
    import nose
    nose.runmodule()
