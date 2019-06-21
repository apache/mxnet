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
import numpy as np
import mxnet as mx
from common import *
from lenet5_common import get_iters


def run_inference(sym, arg_params, aux_params, mnist, all_test_labels, batch_size, use_tensorrt):
    """Run inference with either MXNet or TensorRT"""

    data_size = (batch_size,) + mnist['test_data'].shape[1:]
    type_dict = {'data': 'float32', 'softmax_label': 'float32'}
    if use_tensorrt:
        _sym = sym.get_backend_symbol('TensorRT')
        mx.contrib.tensorrt.init_tensorrt_params(_sym, arg_params, aux_params)
    else:
        _sym = sym
    for k, v in arg_params.items():
        type_dict[k] = v.dtype
    for k, v in aux_params.items():
        type_dict[k] = v.dtype
    executor = _sym.simple_bind(ctx=mx.gpu(0),
                                type_dict=type_dict,
                                data=data_size,
                                softmax_label=(batch_size,),
                                grad_req='null',
                                force_rebind=True)
    executor.copy_params_from(arg_params, aux_params)
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
    """Run LeNet-5 inference comparison between MXNet and TensorRT."""
    check_tensorrt_installation()
    mnist = mx.test_utils.get_mnist()
    num_epochs = 10
    batch_size = 128
    model_name = 'lenet5'
    model_dir = os.getenv("LENET_MODEL_DIR", "/tmp")
    model_file = '%s/%s-symbol.json' % (model_dir, model_name)
    params_file = '%s/%s-%04d.params' % (model_dir, model_name, num_epochs)

    _, _, _, all_test_labels = get_iters(mnist, batch_size)

    # Load serialized MXNet model (model-symbol.json + model-epoch.params)
    sym, arg_params, aux_params = mx.model.load_checkpoint(model_name, num_epochs)

    print("LeNet-5 test")
    print("Running inference in MXNet")
    mx_pct = run_inference(sym, arg_params, aux_params, mnist, all_test_labels,
                           batch_size=batch_size, use_tensorrt=False)

    print("Running inference in MXNet-TensorRT")
    trt_pct = run_inference(sym, arg_params, aux_params, mnist, all_test_labels,
                            batch_size=batch_size, use_tensorrt=True)

    print("MXNet accuracy: %f" % mx_pct)
    print("MXNet-TensorRT accuracy: %f" % trt_pct)

    absolute_accuracy_diff = abs(mx_pct - trt_pct)
    epsilon = 1.01e-2
    assert absolute_accuracy_diff < epsilon, \
        """Absolute diff. between MXNet & TensorRT accuracy (%f) exceeds threshold (%f):
           MXNet = %f, TensorRT = %f""" % (absolute_accuracy_diff, epsilon, mx_pct, trt_pct)


if __name__ == '__main__':
    import nose
    nose.runmodule()
