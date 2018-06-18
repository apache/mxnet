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
#pylint: disable=unused-import
import unittest
#pylint: enable=unused-import
import numpy as np
import mxnet as mx
from ctypes.util import find_library
from mxnet.cuda_utils import get_device_count

assert get_device_count() > 0, "No GPUs available to test TensorRT"

assert find_library('nvinfer') is not None, "Can't find the TensorRT shared library"

def get_use_tensorrt():
    return int(os.environ.get("MXNET_USE_TENSORRT", 0))

def set_use_tensorrt(status = False):
    os.environ["MXNET_USE_TENSORRT"] = str(int(status))

def merge_dicts(*dict_args):
    """Merge arg_params and aux_params to populate shared_buffer"""
    result = {}
    for dictionary in dict_args:
        result.update(dictionary)
    return result

def get_iters(mnist, batch_size):
    """Get MNIST iterators."""
    train_iter = mx.io.NDArrayIter(mnist['train_data'],
                                   mnist['train_label'],
                                   batch_size,
                                   shuffle=True)
    val_iter = mx.io.NDArrayIter(mnist['test_data'], mnist['test_label'], batch_size)
    test_iter = mx.io.NDArrayIter(mnist['test_data'], mnist['test_label'], batch_size)
    all_test_labels = np.array(mnist['test_label'])
    return train_iter, val_iter, test_iter, all_test_labels

def lenet5():
    """LeNet-5 Symbol"""
    #pylint: disable=no-member
    data = mx.sym.Variable('data')
    conv1 = mx.sym.Convolution(data=data, kernel=(5, 5), num_filter=20)
    tanh1 = mx.sym.Activation(data=conv1, act_type="tanh")
    pool1 = mx.sym.Pooling(data=tanh1, pool_type="max",
                           kernel=(2, 2), stride=(2, 2))
    # second conv
    conv2 = mx.sym.Convolution(data=pool1, kernel=(5, 5), num_filter=50)
    tanh2 = mx.sym.Activation(data=conv2, act_type="tanh")
    pool2 = mx.sym.Pooling(data=tanh2, pool_type="max",
                           kernel=(2, 2), stride=(2, 2))
    # first fullc
    flatten = mx.sym.Flatten(data=pool2)
    fc1 = mx.sym.FullyConnected(data=flatten, num_hidden=500)
    tanh3 = mx.sym.Activation(data=fc1, act_type="tanh")
    # second fullc
    fc2 = mx.sym.FullyConnected(data=tanh3, num_hidden=10)
    # loss
    lenet = mx.sym.SoftmaxOutput(data=fc2, name='softmax')
    #pylint: enable=no-member
    return lenet

def train_lenet5(num_epochs, batch_size, train_iter, val_iter, test_iter):
    """train LeNet-5 model on MNIST data"""
    ctx = mx.gpu(0)
    lenet_model = mx.mod.Module(lenet5(), context=ctx)

    lenet_model.fit(train_iter,
                    eval_data=val_iter,
                    optimizer='sgd',
                    optimizer_params={'learning_rate': 0.1, 'momentum': 0.9},
                    eval_metric='acc',
                    batch_end_callback=mx.callback.Speedometer(batch_size, 1),
                    num_epoch=num_epochs)

    # predict accuracy for lenet
    acc = mx.metric.Accuracy()
    lenet_model.score(test_iter, acc)
    accuracy = acc.get()[1]
    assert accuracy > 0.95, "LeNet-5 training accuracy on MNIST was too low"
    return lenet_model

def run_inference(sym, arg_params, aux_params, mnist, all_test_labels, batch_size):
    """Run inference with either MxNet or TensorRT"""

    shared_buffer = merge_dicts(arg_params, aux_params)
    if not get_use_tensorrt():
        shared_buffer = dict([(k, v.as_in_context(mx.gpu(0))) for k, v in shared_buffer.items()])
    executor = sym.simple_bind(ctx=mx.gpu(0),
                               data=(batch_size,) +  mnist['test_data'].shape[1:],
                               softmax_label=(batch_size,),
                               shared_buffer=shared_buffer,
                               grad_req='null',
                               force_rebind=True)

    # Get this value from all_test_labels
    # Also get classes from the dataset
    num_ex = 10000
    all_preds = np.zeros([num_ex, 10])
    test_iter = mx.io.NDArrayIter(mnist['test_data'], mnist['test_label'], batch_size)

    example_ct = 0

    for idx, dbatch in enumerate(test_iter):
        executor.arg_dict["data"][:] = dbatch.data[0]
        executor.forward(is_train=False)
        offset = idx*batch_size
        extent = batch_size if num_ex - offset > batch_size else num_ex - offset
        all_preds[offset:offset+extent, :] = executor.outputs[0].asnumpy()[:extent]
        example_ct += extent

    all_preds = np.argmax(all_preds, axis=1)
    matches = (all_preds[:example_ct] == all_test_labels[:example_ct]).sum()

    percentage = 100.0 * matches / example_ct

    return percentage


def test_tensorrt_inference():
    """Run inference comparison between MxNet and TensorRT.
       This could be used stand-alone or with nosetests."""
    mnist = mx.test_utils.get_mnist()
    num_epochs = 10
    batch_size = 1024
    model_name = 'lenet5'
    model_file = '%s-symbol.json' % model_name
    params_file = '%s-%04d.params' % (model_name, num_epochs)

    _, _, _, all_test_labels = get_iters(mnist, batch_size)

    if not (os.path.exists(model_file) and os.path.exists(params_file)):
        trained_lenet = train_lenet5(num_epochs, batch_size,
                                     *get_iters(mnist, batch_size)[:-1])
        trained_lenet.save_checkpoint(model_name, num_epochs)

    # Load serialized MxNet model (model-symbol.json + model-epoch.params)
    sym, arg_params, aux_params = mx.model.load_checkpoint(model_name, num_epochs)

    print("Running inference in MxNet")
    set_use_tensorrt(False)
    mx_pct = run_inference(sym, arg_params, aux_params, mnist,
                           all_test_labels, batch_size=batch_size)

    print("Running inference in MxNet-TensorRT")
    set_use_tensorrt(True)
    trt_pct = run_inference(sym, arg_params, aux_params, mnist,
                            all_test_labels,  batch_size=batch_size)

    print("MxNet accuracy: %f" % mx_pct)
    print("MxNet-TensorRT accuracy: %f" % trt_pct)

    assert abs(mx_pct - trt_pct) < 1e-2, \
        """Diff. between MxNet & TensorRT accuracy too high:
           MxNet = %f, TensorRT = %f""" % (mx_pct, trt_pct)

if __name__ == '__main__':
    test_tensorrt_inference()

