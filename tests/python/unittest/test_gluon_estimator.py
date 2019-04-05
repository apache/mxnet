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

''' Unit tests for Gluon Estimator '''

import sys
import unittest
import warnings

import mxnet as mx
from mxnet import gluon
from mxnet.gluon import nn
from mxnet.gluon.estimator import Estimator, EventHandler
from nose.tools import assert_raises


def get_model():
    net = nn.Sequential()
    net.add(nn.Dense(4, activation='relu', flatten=False))
    return net


def test_fit():
    ''' test estimator with different train data types '''
    net = get_model()
    num_epochs = 1
    batch_size = 4
    ctx = mx.cpu()
    loss = gluon.loss.L2Loss()
    acc = mx.metric.Accuracy()
    net.initialize(ctx=ctx)
    trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.001})
    est = Estimator(net=net,
                    loss=loss,
                    metrics=acc,
                    trainer=trainer,
                    context=ctx)
    in_data = mx.nd.random.uniform(shape=(10, 3))
    out_data = mx.nd.random.uniform(shape=(10, 4))
    # Input dataloader
    dataset = gluon.data.dataset.ArrayDataset(in_data, out_data)
    train_dataloader = gluon.data.DataLoader(dataset, batch_size=batch_size)
    est.fit(train_data=train_dataloader,
            epochs=num_epochs)

    # Input dataiter
    train_dataiter = mx.io.NDArrayIter(data=in_data, label=out_data, batch_size=batch_size)
    with assert_raises(ValueError):
        est.fit(train_data=train_dataiter,
                epochs=num_epochs)

    # Input NDArray
    with assert_raises(ValueError):
        est.fit(train_data=[in_data, out_data],
                epochs=num_epochs)


def test_validation():
    ''' test different validation data types'''
    net = get_model()
    num_epochs = 1
    batch_size = 4
    ctx = mx.cpu()
    loss = gluon.loss.L2Loss()
    acc = mx.metric.Accuracy()
    net.initialize(ctx=ctx)
    trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.001})
    est = Estimator(net=net,
                    loss=loss,
                    metrics=acc,
                    trainer=trainer,
                    context=ctx)
    in_data = mx.nd.random.uniform(shape=(10, 3))
    out_data = mx.nd.random.uniform(shape=(10, 4))
    # Input dataloader
    dataset = gluon.data.dataset.ArrayDataset(in_data, out_data)
    train_dataloader = gluon.data.DataLoader(dataset, batch_size=batch_size)
    val_dataloader = gluon.data.DataLoader(dataset, batch_size=batch_size)
    est.fit(train_data=train_dataloader,
            val_data=val_dataloader,
            epochs=num_epochs)

    # Input dataiter
    train_dataiter = mx.io.NDArrayIter(data=in_data, label=out_data, batch_size=batch_size)
    val_dataiter = mx.io.NDArrayIter(data=in_data, label=out_data, batch_size=batch_size)
    with assert_raises(ValueError):
        est.fit(train_data=train_dataiter,
                val_data=val_dataiter,
                epochs=num_epochs)
    # Input NDArray
    with assert_raises(ValueError):
        est.fit(train_data=[in_data, out_data],
                val_data=[in_data, out_data],
                epochs=num_epochs)


@unittest.skipIf(sys.version_info.major < 3, 'Test on python 3')
def test_initializer():
    ''' test with no initializer, inconsistent initializer '''
    net = get_model()
    num_epochs = 1
    batch_size = 4
    ctx = mx.cpu()
    in_data = mx.nd.random.uniform(shape=(10, 3))
    out_data = mx.nd.random.uniform(shape=(10, 4))
    dataset = gluon.data.dataset.ArrayDataset(in_data, out_data)
    train_data = gluon.data.DataLoader(dataset, batch_size=batch_size)
    loss = gluon.loss.L2Loss()
    acc = mx.metric.Accuracy()
    # no initializer
    est = Estimator(net=net,
                    loss=loss,
                    metrics=acc,
                    context=ctx)
    est.fit(train_data=train_data,
            epochs=num_epochs)

    # different initializer for net and estimator
    net = get_model()
    net.initialize(mx.init.Xavier(), ctx=ctx)
    trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.001})
    # catch reinit warning
    with warnings.catch_warnings(record=True) as w:
        est = Estimator(net=net,
                        loss=loss,
                        metrics=acc,
                        initializer=mx.init.MSRAPrelu(),
                        trainer=trainer,
                        context=ctx)
        assert 'Network already initialized' in str(w[-1].message)
    est.fit(train_data=train_data,
            epochs=num_epochs)


@unittest.skipIf(sys.version_info.major < 3, 'Test on python 3')
def test_trainer():
    ''' test with no trainer and invalid trainer '''
    net = get_model()
    num_epochs = 1
    batch_size = 4
    ctx = mx.cpu()
    in_data = mx.nd.random.uniform(shape=(10, 3))
    out_data = mx.nd.random.uniform(shape=(10, 4))
    dataset = gluon.data.dataset.ArrayDataset(in_data, out_data)
    train_data = gluon.data.DataLoader(dataset, batch_size=batch_size)
    loss = gluon.loss.L2Loss()
    acc = mx.metric.Accuracy()
    net.initialize(ctx=ctx)
    # input no trainer
    with warnings.catch_warnings(record=True) as w:
        est = Estimator(net=net,
                        loss=loss,
                        metrics=acc,
                        context=ctx)
        assert 'No trainer specified' in str(w[-1].message)
    est.fit(train_data=train_data,
            epochs=num_epochs)

    # input invalid trainer
    trainer = 'sgd'
    with assert_raises(ValueError):
        est = Estimator(net=net,
                        loss=loss,
                        metrics=acc,
                        trainer=trainer,
                        context=ctx)


def test_metric():
    ''' test with no metric, list of metrics, invalid metric '''
    net = get_model()
    num_epochs = 1
    batch_size = 4
    ctx = mx.cpu()
    in_data = mx.nd.random.uniform(shape=(10, 3))
    out_data = mx.nd.random.uniform(shape=(10, 4))
    dataset = gluon.data.dataset.ArrayDataset(in_data, out_data)
    train_data = gluon.data.DataLoader(dataset, batch_size=batch_size)
    loss = gluon.loss.L2Loss()
    net.initialize(ctx=ctx)
    trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.001})
    # input no metric
    est = Estimator(net=net,
                    loss=loss,
                    trainer=trainer,
                    context=ctx)
    est.fit(train_data=train_data,
            epochs=num_epochs)
    # input list of metrics
    metrics = [mx.metric.Accuracy(), mx.metric.Accuracy()]
    est = Estimator(net=net,
                    loss=loss,
                    metrics=metrics,
                    trainer=trainer,
                    context=ctx)
    est.fit(train_data=train_data,
            epochs=num_epochs)
    # input invalid metric
    with assert_raises(ValueError):
        est = Estimator(net=net,
                        loss=loss,
                        metrics='acc',
                        trainer=trainer,
                        context=ctx)
    # test default metric
    loss = gluon.loss.SoftmaxCrossEntropyLoss()
    est = Estimator(net=net,
                    loss=loss,
                    trainer=trainer,
                    context=ctx)
    assert isinstance(est.train_metrics[0], mx.metric.Accuracy)


def test_loss():
    ''' test with invalid loss '''
    net = get_model()
    ctx = mx.cpu()
    acc = mx.metric.Accuracy()
    net.initialize(ctx=ctx)
    trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.001})
    # input invalid loss
    with assert_raises(ValueError):
        est = Estimator(net=net,
                        loss='mse',
                        metrics=acc,
                        trainer=trainer,
                        context=ctx)


def test_context():
    ''' test with no context, list of context, invalid context '''
    net = get_model()
    loss = gluon.loss.L2Loss()
    metrics = mx.metric.Accuracy()
    # input no context
    est = Estimator(net=net,
                    loss=loss,
                    metrics=metrics)
    # input list of context
    gpus = mx.context.num_gpus()
    ctx = [mx.gpu(i) for i in gpus] if gpus > 0 else [mx.cpu()]
    net = get_model()
    est = Estimator(net=net,
                    loss=loss,
                    metrics=metrics,
                    context=ctx)
    # input invalid context
    with assert_raises(ValueError):
        est = Estimator(net=net,
                        loss=loss,
                        metrics=metrics,
                        context='cpu')


def test_categorize_handlers():
    class CustomHandler1(EventHandler):
        def __init__(self):
            super(CustomHandler1, self).__init__()

        def train_begin(self):
            print("custom train begin")

    class CustomHandler2(EventHandler):
        def __init__(self):
            super(CustomHandler2, self).__init__()

        def epoch_begin(self):
            print("custom epoch begin")

        def batch_begin(self):
            print("custom batch begin")

        def train_end(self):
            print("custom train end")

    class CustomHandler3(EventHandler):
        def __init__(self):
            super(CustomHandler3, self).__init__()

        def epoch_begin(self):
            print("custom epoch begin")

        def batch_begin(self):
            print("custom batch begin")

        def batch_end(self):
            print("custom batch end")

        def train_end(self):
            print("custom train end")

    net = nn.Sequential()
    net.add(nn.Dense(10))
    loss = gluon.loss.SoftmaxCrossEntropyLoss()
    est = Estimator(net, loss=loss)
    event_handlers = [CustomHandler1(), CustomHandler2(), CustomHandler3()]
    train_begin, epoch_begin, batch_begin, \
    batch_end, epoch_end, train_end = est._categorize_handlers(event_handlers)
    assert len(train_begin) == 1
    assert len(epoch_begin) == 2
    assert len(batch_begin) == 2
    assert len(batch_end) == 1
    assert len(train_end) == 2
