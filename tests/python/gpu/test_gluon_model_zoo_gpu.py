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
import os
import unittest
curr_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
sys.path.insert(0, os.path.join(curr_path, '../unittest'))
from common import setup_module, with_seed, teardown

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

VAL_DATA='data/val-5k-256.rec'
def download_data():
    return mx.test_utils.download(
        'http://data.mxnet.io/data/val-5k-256.rec', VAL_DATA)

@with_seed()
def test_inference():
    all_models = ['resnet50_v1', 'vgg19_bn', 'alexnet', #'inceptionv3',
                  'densenet201', 'squeezenet1.0', 'mobilenet0.25']

    batch_size = 10
    download_data()
    for model_name in all_models:
        eprint('testing inference on %s'%model_name)

        data_shape = (3, 224, 224) if 'inception' not in model_name else (3, 299, 299)
        dataIter = mx.io.ImageRecordIter(
            path_imgrec        = VAL_DATA,
            label_width        = 1,
            preprocess_threads = 1,
            batch_size         = batch_size,
            data_shape         = data_shape,
            label_name         = 'softmax_label',
            rand_crop          = False,
            rand_mirror        = False)
        data_batch = dataIter.next()
        data = data_batch.data[0]
        label = data_batch.label[0]
        gpu_data = data.as_in_context(mx.gpu())
        gpu_label = label.as_in_context(mx.gpu())

        # This is to create a model and run the model once to initialize
        # all parameters.
        cpu_model = get_model(model_name)
        cpu_model.collect_params().initialize(ctx=mx.cpu())
        cpu_model(mx.nd.array(data, ctx=mx.cpu()))
        gpu_model = get_model(model_name)
        gpu_model.collect_params().initialize(ctx=mx.gpu())
        gpu_model(mx.nd.array(data, ctx=mx.gpu()))

        # Force the two models have the same parameters.
        cpu_params = cpu_model.collect_params()
        gpu_params = gpu_model.collect_params()
        for k in cpu_params.keys():
            k = k.replace(cpu_params.prefix, '')
            cpu_param = cpu_params.get(k)
            gpu_param = gpu_params.get(k)
            gpu_param.set_data(cpu_param.data().as_in_context(mx.gpu()))

        for i in range(5):
            # Run inference.
            with autograd.record(train_mode=False):
                cpu_out = cpu_model(mx.nd.array(data, ctx=mx.cpu()))
                gpu_out = gpu_model(gpu_data)
            out = cpu_out.asnumpy()
            max_val = np.max(np.abs(out))
            gpu_max_val = np.max(np.abs(gpu_out.asnumpy()))
            eprint(model_name + ": CPU " + str(max_val) + ", GPU " + str(gpu_max_val))
            assert_almost_equal(out / max_val, gpu_out.asnumpy() / max_val, rtol=1e-3, atol=1e-3)

def get_nn_model(name):
    if "densenet" in name:
        return get_model(name, dropout=0)
    else:
        return get_model(name)

# Seed 1521019752 produced a failure on the Py2 MKLDNN-GPU CI runner
# on 2/16/2018 that was not reproducible.  Problem could be timing related or
# based on non-deterministic algo selection.
@with_seed()
def test_training():
    # We use network models without dropout for testing.
    # TODO(zhengda) mobilenet can't pass this test even without MKLDNN.
    all_models = ['resnet18_v1', 'densenet121']

    batch_size = 10
    label = mx.nd.random.uniform(low=0, high=10, shape=(batch_size)).astype('int32')

    download_data()
    dataIter = mx.io.ImageRecordIter(
        path_imgrec        = VAL_DATA,
        label_width        = 1,
        preprocess_threads = 1,
        batch_size         = batch_size,
        data_shape         = (3, 224, 224),
        label_name         = 'softmax_label',
        rand_crop          = False,
        rand_mirror        = False)
    data_batch = dataIter.next()
    data = data_batch.data[0]
    label = data_batch.label[0]
    gpu_data = data.as_in_context(mx.gpu())
    gpu_label = label.as_in_context(mx.gpu())
    softmax_cross_entropy = mx.gluon.loss.SoftmaxCrossEntropyLoss()

    for model_name in all_models:
        eprint('testing %s'%model_name)
        #data = mx.nd.random.uniform(shape=(100, 3, 224, 224))

        # This is to create a model and run the model once to initialize
        # all parameters.
        cpu_model = get_nn_model(model_name)
        cpu_model.collect_params().initialize(ctx=mx.cpu())
        cpu_model(mx.nd.array(data, ctx=mx.cpu()))
        gpu_model = get_nn_model(model_name)
        gpu_model.collect_params().initialize(ctx=mx.gpu())
        gpu_model(mx.nd.array(data, ctx=mx.gpu()))

        # Force the two models have the same parameters.
        cpu_params = cpu_model.collect_params()
        gpu_params = gpu_model.collect_params()
        for k in cpu_params.keys():
            k = k.replace(cpu_params.prefix, '')
            cpu_param = cpu_params.get(k)
            gpu_param = gpu_params.get(k)
            gpu_param.set_data(cpu_param.data().as_in_context(mx.gpu()))

        cpu_trainer = mx.gluon.Trainer(cpu_params, 'sgd', {'learning_rate': 0.1})
        gpu_trainer = mx.gluon.Trainer(gpu_params, 'sgd', {'learning_rate': 0.1})

        # Run forward and backward once.
        with autograd.record():
            cpu_out = cpu_model(mx.nd.array(data, ctx=mx.cpu()))
            gpu_out = gpu_model(gpu_data)
            cpu_loss = softmax_cross_entropy(cpu_out, label)
            gpu_loss = softmax_cross_entropy(gpu_out, gpu_label)
        max_val = np.max(np.abs(cpu_out.asnumpy()))
        gpu_max_val = np.max(np.abs(gpu_out.asnumpy()))
        eprint(model_name + ": CPU " + str(max_val) + ", GPU " + str(gpu_max_val))
        assert_almost_equal(cpu_out.asnumpy() / max_val, gpu_out.asnumpy() / max_val, rtol=1e-3, atol=1e-3)
        cpu_loss.backward()
        gpu_loss.backward()
        cpu_trainer.step(batch_size)
        gpu_trainer.step(batch_size)

        # Compare the parameters of the two models.
        start_test = False
        for k in cpu_params.keys():
            print(k)
            if "stage3" in k:
                start_test = True
            if (start_test):
                k = k.replace(cpu_params.prefix, '')
                cpu_param = cpu_params.get(k)
                gpu_param = gpu_params.get(k)
                assert_almost_equal(cpu_param.data().asnumpy(), gpu_param.data().asnumpy(),
                        rtol=1e-3, atol=1e-3)

if __name__ == '__main__':
    import nose
    nose.runmodule()
