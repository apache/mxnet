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
import mxnet as mx
from lenet5_common import get_iters


def lenet5():
    """LeNet-5 Symbol"""
    #pylint: disable=no-member
    data = mx.sym.Variable('data')
    data = mx.sym.Cast(data, 'float16')
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
    fc2 = mx.sym.Cast(fc2, 'float32')
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


if __name__ == '__main__':
    num_epochs = 10
    batch_size = 128
    model_name = 'lenet5'
    model_dir = os.getenv("LENET_MODEL_DIR", "/tmp")
    model_file = '%s/%s-symbol.json' % (model_dir, model_name)
    params_file = '%s/%s-%04d.params' % (model_dir, model_name, num_epochs)

    if not (os.path.exists(model_file) and os.path.exists(params_file)):
        mnist = mx.test_utils.get_mnist()

        _, _, _, all_test_labels = get_iters(mnist, batch_size)

        trained_lenet = train_lenet5(num_epochs, batch_size,
                                    *get_iters(mnist, batch_size)[:-1])
        trained_lenet.save_checkpoint(model_name, num_epochs)
