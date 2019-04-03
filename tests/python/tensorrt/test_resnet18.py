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
    return img/255.0


def test_tensorrt_resnet18_feature_vect():
    print("downloading sample input")
    input_data = get_image(url)
    gluon_resnet18 = vision.resnet18_v2(pretrained=True)
    gluon_resnet18.hybridize()
    gluon_resnet18.forward(input_data)
    gluon_resnet18.export(model_file_name)
    sym, arg_params, aux_params = mx.model.load_checkpoint(model_file_name, 0)

    os.environ['MXNET_USE_TENSORRT'] = '0'
    executor = sym.simple_bind(ctx=mx.gpu(), data=batch_shape, grad_req='null', force_rebind=True)
    executor.copy_params_from(arg_params, aux_params)
    y = executor.forward(is_train=False, data=input_data)

    os.environ['MXNET_USE_TENSORRT'] = '1'
    all_params = arg_params
    all_params.update(aux_params)
    executor = mx.contrib.tensorrt.tensorrt_bind(sym, ctx=mx.gpu(), all_params=all_params, data=batch_shape,
                                                 grad_req='null', force_rebind=True)
    y_trt = executor.forward(is_train=False, data=input_data)

    no_trt_output = y[0].asnumpy()[0]
    trt_output = y_trt[0].asnumpy()[0]
    assert_almost_equal(no_trt_output, trt_output, 1e-4, 1e-4)


if __name__ == '__main__':
    import nose

    nose.runmodule()
