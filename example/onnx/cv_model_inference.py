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

import os
import json
import urllib.request
import mxnet as mx
import numpy as np
import gluoncv
import onnxruntime
from urllib.parse import urlparse
from mxnet.gluon.data.vision import transforms

def preprocess_image(imgfile, resize_short=256, crop_size=224,
                   mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    # load image
    img_data = mx.image.imread(imgfile).astype('float32')
    # normalization and standerdization
    transform_fn = transforms.Compose([
        transforms.Resize(resize_short, keep_ratio=True),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    # expand batch dimension
    res = transform_fn(img_data).expand_dims(0)
    # convert mx ndarray to np ndarray for onnxruntime
    res = res.asnumpy()
    return res

# molde path prefix
prefix = './resnet50_v2'
# input shape and type
in_shape = (1, 3, 224, 224)
in_dtype = 'float32'
# download model
gluon_model = gluoncv.model_zoo.get_model('resnet50_v2', pretrained=True)
gluon_model.hybridize()
# forward with dummy input and save model
dummy_input = mx.nd.zeros(in_shape, dtype=in_dtype)
gluon_model.forward(dummy_input)
gluon_model.export(prefix, 0)

# mxnet model symbol file
mx_sym = prefix + '-symbol.json'
# mxnet model params file
mx_params = prefix + '-0000.params'
# onnx model file that will be exported
onnx_file = prefix + '.onnx'
# list of shape for all inputs
in_shapes = [in_shape]
# list of data type for all inputs
in_types = [in_dtype]
# export onnx model
mx.onnx.export_model(mx_sym, mx_params, in_shapes, in_types, onnx_file)

# # example for dynamic input shape (optional)
# # None indicating dynamic shape at a certain dimension
# dynamic_input_shapes = [((None, 3, 224, 224))]
# mx.onnx.export_model(mx_sym, mx_params, in_shapes, in_types, onnx_file,
#                      dynamic=True, dynamic_input_shapes=dynamic_input_shapes)

# download and process the input image
img_dir = './images'
img_url = 'https://github.com/apache/incubator-mxnet-ci/raw/master/test-data/images/car.jpg'
fname = os.path.join(img_dir, os.path.basename(urlparse(img_url).path))
mx.test_utils.download(img_url, fname=fname)
img_data = preprocess_image(fname)

# create onnxruntime session using the onnx model file
ses_opt = onnxruntime.SessionOptions()
ses_opt.log_severity_level = 3
session = onnxruntime.InferenceSession(onnx_file, ses_opt)
input_name = session.get_inputs()[0].name

# run onnx inference
onnx_result = session.run([], {input_name: img_data})[0]
idx = np.argmax(onnx_result, axis=1).astype('int')[0]

# post processing: map class index to class name
url = 'https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json'
urllib.request.urlretrieve(url, './imagenet_class_index.json')
class_idx = json.load(open('imagenet_class_index.json'))
idx2label = [class_idx[str(k)][1] for k in range(len(class_idx))]
print(idx2label[idx])
