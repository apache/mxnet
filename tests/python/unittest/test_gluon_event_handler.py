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
from common import TemporaryDirectory
from mxnet import nd
from mxnet.gluon import nn, loss
from mxnet.gluon.contrib.estimator import estimator, event_handler


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
    with TemporaryDirectory() as tmpdir:
        model_prefix = 'test_net'
        file_path = os.path.join(tmpdir, model_prefix)
        test_data = _get_test_data()

        net = _get_test_network()
        ce_loss = loss.SoftmaxCrossEntropyLoss()
        acc = mx.metric.Accuracy()
        est = estimator.Estimator(net, loss=ce_loss, metrics=acc)
        checkpoint_handler = [event_handler.CheckpointHandler(model_dir=tmpdir,
                                                              model_prefix=model_prefix,
                                                              monitor=acc,
                                                              save_best=True)]
        est.fit(test_data, event_handlers=checkpoint_handler, epochs=1)
        assert os.path.isfile(file_path + '-best.params')
        assert os.path.isfile(file_path + '-best.states')
        assert os.path.isfile(file_path + '-epoch0.params')
        assert os.path.isfile(file_path + '-epoch0.states')

        model_prefix = 'test_batch'
        file_path = os.path.join(tmpdir, model_prefix)
        checkpoint_handler = [event_handler.CheckpointHandler(model_dir=tmpdir,
                                                              model_prefix=model_prefix,
                                                              epoch_period=None,
                                                              batch_period=1,
                                                              max_checkpoints=2)]
        est.fit(test_data, event_handlers=checkpoint_handler, epochs=3)
        assert not os.path.isfile(file_path + 'best.params')
        assert not os.path.isfile(file_path + 'best.states')
        assert not os.path.isfile(file_path + '-batch0.params')
        assert not os.path.isfile(file_path + '-batch0.states')
        assert os.path.isfile(file_path + '-batch1.params')
        assert os.path.isfile(file_path + '-batch1.states')
        assert os.path.isfile(file_path + '-batch2.params')
        assert os.path.isfile(file_path + '-batch2.states')



def test_early_stopping():
    test_data = _get_test_data()

    mode = 'max'
    patience = 0

    net = _get_test_network()
    ce_loss = loss.SoftmaxCrossEntropyLoss()
    acc = mx.metric.Accuracy()
    est = estimator.Estimator(net, loss=ce_loss, metrics=acc)
    early_stopping = [event_handler.EarlyStoppingHandler(monitor=acc,
                                                         patience=patience,
                                                         mode=mode)]
    est.fit(test_data, event_handlers=early_stopping, epochs=3)

    mode = 'auto'
    patience = 2
    early_stopping = [event_handler.EarlyStoppingHandler(monitor=acc,
                                                         patience=patience,
                                                         mode=mode)]
    est.fit(test_data, event_handlers=early_stopping, epochs=1)


def test_logging():
    with TemporaryDirectory() as tmpdir:
        test_data = _get_test_data()
        file_name = 'test_log'
        output_dir = os.path.join(tmpdir, file_name)

        net = _get_test_network()
        ce_loss = loss.SoftmaxCrossEntropyLoss()
        acc = mx.metric.Accuracy()
        est = estimator.Estimator(net, loss=ce_loss, metrics=acc)
        train_metrics, val_metrics = est.prepare_loss_and_metrics()
        logging_handler = [event_handler.LoggingHandler(file_name=file_name,
                                                        file_location=tmpdir,
                                                        train_metrics=train_metrics,
                                                        val_metrics=val_metrics)]
        est.fit(test_data, event_handlers=logging_handler, epochs=1)
        assert os.path.isfile(output_dir)
