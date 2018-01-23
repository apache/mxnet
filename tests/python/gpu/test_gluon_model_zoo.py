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

from __future__ import print_function
import mxnet as mx
import numpy as np
import copy
from mxnet import autograd
from mxnet.gluon.model_zoo.vision import get_model
from mxnet.test_utils import assert_almost_equal
import sys

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

VAL_DATA='data/val-5k-256.rec'
def download_data():
    return mx.test_utils.download(
        'http://data.mxnet.io/data/val-5k-256.rec', VAL_DATA)

def test_models():
    all_models = ['resnet18_v1', 'densenet121', 'mobilenet1.0']

    n = 10
    label = mx.nd.random.uniform(low=0, high=10, shape=(n)).astype('int32')

    download_data()
    dataIter = mx.io.ImageRecordIter(
        path_imgrec        = VAL_DATA,
        label_width        = 1,
        preprocess_threads = 1,
        batch_size         = n,
        data_shape         = (3, 224, 224),
        label_name         = 'softmax_label',
        rand_crop          = False,
        rand_mirror        = False)
    data_batch = dataIter.next()
    softmax_cross_entropy = mx.gluon.loss.SoftmaxCrossEntropyLoss()

    for model_name in all_models:
        eprint('testing forward for %s'%model_name)
        data = data_batch.data[0]
        label = data_batch.label[0]
        #data = mx.nd.random.uniform(shape=(100, 3, 224, 224))

        cpu_model = get_model(model_name)
        cpu_model.collect_params().initialize(ctx=mx.cpu())
        cpu_out = cpu_model(mx.nd.array(data, ctx=mx.cpu()))
        gpu_model = get_model(model_name)
        gpu_model.collect_params().initialize(ctx=mx.gpu())

        cpu_params = cpu_model.collect_params()
        gpu_params = gpu_model.collect_params()
        for k in cpu_params.keys():
            k = k.replace(cpu_params.prefix, '')
            cpu_param = cpu_params.get(k)
            gpu_param = gpu_params.get(k)
            gpu_param.set_data(cpu_param.data().as_in_context(mx.gpu()))

        with autograd.record():
            cpu_out = cpu_model(mx.nd.array(data, ctx=mx.cpu()))
            gpu_out = gpu_model(mx.nd.array(data, ctx=mx.gpu()))
            cpu_loss = softmax_cross_entropy(cpu_out, label)
        assert_almost_equal(cpu_out.asnumpy() / cpu_out.asnumpy(), gpu_out.asnumpy() / cpu_out.asnumpy(),
                rtol=1e-1, atol=1e-1)
        #cpu_loss.backward()


if __name__ == '__main__':
    import nose
    nose.runmodule()
