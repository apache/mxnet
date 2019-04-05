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
import tempfile
import mxnet as mx
from mxnet import nd
from mxnet.gluon import nn, loss
from mxnet.gluon.estimator import estimator, event_handler

def _get_test_network():
    net = nn.Sequential()
    net.add(nn.Dense(128, activation='relu', in_units=100, flatten=False),
              nn.Dense(64, activation='relu', in_units=128),
              nn.Dense(10, activation='relu', in_units=64))
    return net

def _get_test_data():
    data = nd.ones((32, 100))
    label = nd.random.randint(0, 10, (32, 1))
    data_arr = mx.gluon.data.dataset.ArrayDataset(data, label)
    return mx.gluon.data.DataLoader(data_arr, batch_size=32)


def test_checkpoint_handler():
    tmpdir = tempfile.mkdtemp()
    file_path = os.path.join(tmpdir, "model.params")
    test_data  = _get_test_data()

    save_best_only = False
    mode = 'auto'

    net = _get_test_network()
    ce_loss = loss.SoftmaxCrossEntropyLoss()
    acc = mx.metric.Accuracy()
    est = estimator.Estimator(net, loss=ce_loss, metrics=acc)
    checkpoint_handler = [event_handler.CheckpointHandler(file_path,
                                                          save_best_only=save_best_only,
                                                          mode=mode)]
    est.fit(test_data, event_handlers=checkpoint_handler, epochs=1)
    assert os.path.isfile(file_path)
    os.remove(file_path)

def test_early_stopping():
    test_data = _get_test_data()

    mode = 'max'
    monitor = 'train_accuracy'
    patience = 0

    net = _get_test_network()
    ce_loss = loss.SoftmaxCrossEntropyLoss()
    acc = mx.metric.Accuracy()
    est = estimator.Estimator(net, loss=ce_loss, metrics=acc)
    early_stopping = [event_handler.EarlyStoppingHandler(monitor,
                                                         patience=patience,
                                                         mode=mode)]
    est.fit(test_data, event_handlers=early_stopping, epochs=3)

    mode = 'auto'
    monitor = 'train_accuracy'
    patience = 2
    early_stopping = [event_handler.EarlyStoppingHandler(monitor,
                                                         patience=patience,
                                                          mode=mode)]
    est.fit(test_data, event_handlers=early_stopping, epochs=1)

def test_logging():
    tmpdir = tempfile.mkdtemp()
    test_data = _get_test_data()
    file_name = 'test_log'
    output_dir = os.path.join(tmpdir, file_name)

    net = _get_test_network()
    ce_loss = loss.SoftmaxCrossEntropyLoss()
    acc = mx.metric.Accuracy()
    est = estimator.Estimator(net, loss=ce_loss, metrics=acc)
    logging_handler = [event_handler.LoggingHandler(file_name=file_name, file_location=tmpdir)]
    est.fit(test_data, event_handlers=logging_handler, epochs=1)
    assert os.path.isfile(output_dir)
    os.remove(output_dir)