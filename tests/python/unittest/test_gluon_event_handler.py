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
from common import TemporaryDirectory
from mxnet import nd
from mxnet.gluon import nn, loss
from mxnet.gluon.contrib.estimator import estimator, event_handler


def _get_test_network(net=nn.Sequential()):
    net.add(nn.Dense(128, activation='relu', flatten=False),
            nn.Dense(64, activation='relu'),
            nn.Dense(10, activation='relu'))
    return net


def _get_test_data():
    data = nd.ones((32, 100))
    label = nd.zeros((32, 1))
    data_arr = mx.gluon.data.dataset.ArrayDataset(data, label)
    return mx.gluon.data.DataLoader(data_arr, batch_size=8)


def test_checkpoint_handler():
    with TemporaryDirectory() as tmpdir:
        model_prefix = 'test_epoch'
        file_path = os.path.join(tmpdir, model_prefix)
        test_data = _get_test_data()

        net = _get_test_network()
        ce_loss = loss.SoftmaxCrossEntropyLoss()
        acc = mx.metric.Accuracy()
        est = estimator.Estimator(net, loss=ce_loss, metrics=acc)
        checkpoint_handler = event_handler.CheckpointHandler(model_dir=tmpdir,
                                                             model_prefix=model_prefix,
                                                             monitor=acc,
                                                             save_best=True,
                                                             epoch_period=1)
        est.fit(test_data, event_handlers=[checkpoint_handler], epochs=1)
        assert checkpoint_handler.current_epoch == 1
        assert checkpoint_handler.current_batch == 4
        assert os.path.isfile(file_path + '-best.params')
        assert os.path.isfile(file_path + '-best.states')
        assert os.path.isfile(file_path + '-epoch0batch4.params')
        assert os.path.isfile(file_path + '-epoch0batch4.states')

        model_prefix = 'test_batch'
        file_path = os.path.join(tmpdir, model_prefix)
        net = _get_test_network(nn.HybridSequential())
        net.hybridize()
        est = estimator.Estimator(net, loss=ce_loss, metrics=acc)
        checkpoint_handler = event_handler.CheckpointHandler(model_dir=tmpdir,
                                                             model_prefix=model_prefix,
                                                             epoch_period=None,
                                                             batch_period=2,
                                                             max_checkpoints=2)
        est.fit(test_data, event_handlers=[checkpoint_handler], batches=10)
        assert checkpoint_handler.current_batch == 10
        assert checkpoint_handler.current_epoch == 3
        assert not os.path.isfile(file_path + 'best.params')
        assert not os.path.isfile(file_path + 'best.states')
        assert not os.path.isfile(file_path + '-epoch0batch0.params')
        assert not os.path.isfile(file_path + '-epoch0batch0.states')
        assert os.path.isfile(file_path + '-symbol.json')
        assert os.path.isfile(file_path + '-epoch1batch7.params')
        assert os.path.isfile(file_path + '-epoch1batch7.states')
        assert os.path.isfile(file_path + '-epoch2batch9.params')
        assert os.path.isfile(file_path + '-epoch2batch9.states')

def test_resume_checkpoint():
    with TemporaryDirectory() as tmpdir:
        model_prefix = 'test_net'
        file_path = os.path.join(tmpdir, model_prefix)
        test_data = _get_test_data()

        net = _get_test_network()
        ce_loss = loss.SoftmaxCrossEntropyLoss()
        acc = mx.metric.Accuracy()
        est = estimator.Estimator(net, loss=ce_loss, metrics=acc)
        checkpoint_handler = event_handler.CheckpointHandler(model_dir=tmpdir,
                                                             model_prefix=model_prefix,
                                                             monitor=acc,
                                                             max_checkpoints=1)
        est.fit(test_data, event_handlers=[checkpoint_handler], epochs=2)
        assert os.path.isfile(file_path + '-epoch1batch8.params')
        assert os.path.isfile(file_path + '-epoch1batch8.states')
        checkpoint_handler = event_handler.CheckpointHandler(model_dir=tmpdir,
                                                             model_prefix=model_prefix,
                                                             monitor=acc,
                                                             max_checkpoints=1,
                                                             resume_from_checkpoint=True)
        est.fit(test_data, event_handlers=[checkpoint_handler], epochs=5)
        # should only continue to train 3 epochs and last checkpoint file is epoch4
        assert est.max_epoch == 3
        assert os.path.isfile(file_path + '-epoch4batch20.states')


def test_early_stopping():
    test_data = _get_test_data()

    net = _get_test_network()
    ce_loss = loss.SoftmaxCrossEntropyLoss()
    acc = mx.metric.Accuracy()
    est = estimator.Estimator(net, loss=ce_loss, metrics=acc)
    early_stopping = event_handler.EarlyStoppingHandler(monitor=acc,
                                                        patience=0,
                                                        mode='min')
    est.fit(test_data, event_handlers=[early_stopping], epochs=5)
    assert early_stopping.current_epoch == 2
    assert early_stopping.stopped_epoch == 1

    early_stopping = event_handler.EarlyStoppingHandler(monitor=acc,
                                                        patience=2,
                                                        mode='auto')
    est.fit(test_data, event_handlers=[early_stopping], epochs=1)
    assert early_stopping.current_epoch == 1


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
        logging_handler = event_handler.LoggingHandler(file_name=file_name,
                                                       file_location=tmpdir,
                                                       train_metrics=train_metrics,
                                                       val_metrics=val_metrics)
        est.fit(test_data, event_handlers=[logging_handler], epochs=3)
        assert logging_handler.batch_index == 0
        assert logging_handler.current_epoch == 3
        assert os.path.isfile(output_dir)


def test_custom_handler():
    class CustomStopHandler(event_handler.TrainBegin,
                            event_handler.BatchEnd,
                            event_handler.EpochEnd):
        def __init__(self, batch_stop=None, epoch_stop=None):
            self.batch_stop = batch_stop
            self.epoch_stop = epoch_stop
            self.num_batch = 0
            self.num_epoch = 0
            self.stop_training = False

        def train_begin(self, estimator, *args, **kwargs):
            self.num_batch = 0
            self.num_epoch = 0

        def batch_end(self, estimator, *args, **kwargs):
            self.num_batch += 1
            if self.num_batch == self.batch_stop:
                self.stop_training = True
            return self.stop_training

        def epoch_end(self, estimator, *args, **kwargs):
            self.num_epoch += 1
            if self.num_epoch == self.epoch_stop:
                self.stop_training = True
            return self.stop_training

    # total data size is 32, batch size is 8
    # 4 batch per epoch
    test_data = _get_test_data()
    net = _get_test_network()
    ce_loss = loss.SoftmaxCrossEntropyLoss()
    acc = mx.metric.Accuracy()
    est = estimator.Estimator(net, loss=ce_loss, metrics=acc)
    custom_handler = CustomStopHandler(3, 2)
    est.fit(test_data, event_handlers=[custom_handler], epochs=3)
    assert custom_handler.num_batch == 3
    assert custom_handler.num_epoch == 1
    custom_handler = CustomStopHandler(100, 5)
    est.fit(test_data, event_handlers=[custom_handler], epochs=10)
    assert custom_handler.num_batch == 5 * 4
    assert custom_handler.num_epoch == 5
