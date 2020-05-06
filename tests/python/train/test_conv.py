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

# pylint: skip-file
import sys
import mxnet as mx
from mxnet.test_utils import get_mnist_ubyte
import numpy as np
import os, pickle, gzip, argparse
import logging
CURR_PATH = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
sys.path.insert(0, os.path.join(CURR_PATH, '../unittest'))
from common import retry

@retry(3)
def test_mnist(tmpdir):
    def get_model(use_gpu):
        # symbol net
        data = mx.symbol.Variable('data')
        conv1= mx.symbol.Convolution(data = data, name='conv1', num_filter=32, kernel=(3,3), stride=(2,2))
        bn1 = mx.symbol.BatchNorm(data = conv1, name="bn1")
        act1 = mx.symbol.Activation(data = bn1, name='relu1', act_type="relu")
        mp1 = mx.symbol.Pooling(data = act1, name = 'mp1', kernel=(2,2), stride=(2,2), pool_type='max')

        conv2= mx.symbol.Convolution(data = mp1, name='conv2', num_filter=32, kernel=(3,3), stride=(2,2))
        bn2 = mx.symbol.BatchNorm(data = conv2, name="bn2")
        act2 = mx.symbol.Activation(data = bn2, name='relu2', act_type="relu")
        mp2 = mx.symbol.Pooling(data = act2, name = 'mp2', kernel=(2,2), stride=(2,2), pool_type='max')


        fl = mx.symbol.Flatten(data = mp2, name="flatten")
        fc2 = mx.symbol.FullyConnected(data = fl, name='fc2', num_hidden=10)
        softmax = mx.symbol.SoftmaxOutput(data = fc2, name = 'sm')

        num_epoch = 1
        ctx=mx.gpu() if use_gpu else mx.cpu()
        model = mx.model.FeedForward(softmax, ctx,
                                     num_epoch=num_epoch,
                                     learning_rate=0.1, wd=0.0001,
                                     momentum=0.9)
        return model

    def get_iters():
        # check data
        path = str(tmpdir)
        get_mnist_ubyte(path)

        batch_size = 100
        train_dataiter = mx.io.MNISTIter(
                image=os.path.join(path, 'train-images-idx3-ubyte'),
                label=os.path.join(path, 'train-labels-idx1-ubyte'),
                data_shape=(1, 28, 28),
                label_name='sm_label',
                batch_size=batch_size, shuffle=True, flat=False, silent=False, seed=10)
        val_dataiter = mx.io.MNISTIter(
                image=os.path.join(path, 't10k-images-idx3-ubyte'),
                label=os.path.join(path, 't10k-labels-idx1-ubyte'),
                data_shape=(1, 28, 28),
                label_name='sm_label',
                batch_size=batch_size, shuffle=True, flat=False, silent=False)
        return  train_dataiter, val_dataiter

    iters = get_iters()

    def exec_mnist(model, train_dataiter, val_dataiter):
        # print logging by default
        logging.basicConfig(level=logging.DEBUG)
        console = logging.StreamHandler()
        console.setLevel(logging.DEBUG)
        logging.getLogger('').addHandler(console)

        model.fit(X=train_dataiter,
                  eval_data=val_dataiter)
        logging.info('Finish fit...')
        prob = model.predict(val_dataiter)
        logging.info('Finish predict...')
        val_dataiter.reset()
        y = np.concatenate([batch.label[0].asnumpy() for batch in val_dataiter]).astype('int')
        py = np.argmax(prob, axis=1)
        acc1 = float(np.sum(py == y)) / len(y)
        logging.info('final accuracy = %f', acc1)
        assert(acc1 > 0.9)

    exec_mnist(get_model(False), iters[0], iters[1])
