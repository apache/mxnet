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

import mxnet as mx

import sys
import os
curr_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
sys.path.insert(0, os.path.join(curr_path, '../unittest'))
from common import setup_module, with_seed

# This script is testing the efficiency of LARS
# We are training LeNet-5 at batch-size 8000 in 10 epochs above 98% accuracy
# Which is not doable with simple SGD + momentum (from what have been tested so far)

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

@with_seed()
def test_lars():
    num_epochs = 10
    batch_size = 8000
    mnist = mx.test_utils.get_mnist()
    train_iter = mx.io.NDArrayIter(mnist['train_data'],
                                   mnist['train_label'],
                                   batch_size,
                                   shuffle=True)
    test_iter = mx.io.NDArrayIter(mnist['test_data'], mnist['test_label'], batch_size)
    ctx = mx.gpu(0)
    lenet_model = mx.mod.Module(lenet5(), context=ctx)
    warmup_epochs = 1
    epoch_it = int(train_iter.num_data / batch_size)
    # LARS works best with Polynomial scheduler and warmup
    base_lr = 0.01
    optimizer_params={
        'learning_rate': base_lr,
        'lr_scheduler': mx.lr_scheduler.PolyScheduler(base_lr=base_lr,
                                                      max_update=epoch_it * num_epochs,
                                                      warmup_steps=epoch_it * warmup_epochs),
        'momentum': 0.9,
        'eta': 14.,
      }
    lenet_model.fit(train_iter,
                    eval_data=test_iter,
                    optimizer='lars',
                    optimizer_params=optimizer_params,
                    eval_metric='acc',
                    num_epoch=num_epochs)

    # predict accuracy for lenet
    acc = mx.metric.Accuracy()
    lenet_model.score(test_iter, acc)
    accuracy = acc.get()[1]
    assert accuracy > 0.98, "LeNet-5 training accuracy on MNIST was too low"

if __name__ == '__main__':
    import nose
    nose.runmodule()
