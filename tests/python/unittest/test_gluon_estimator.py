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

import warnings
from nose.tools import assert_raises
import mxnet as mx
from mxnet import gluon
from mxnet.gluon import nn
from mxnet.gluon.estimator import estimator
from common import setup_module, with_seed, assertRaises

def get_model():
    net = nn.Sequential()
    net.add(nn.Dense(4, activation='tanh', flatten=False))
    return net

@with_seed()
def test_fit():
    net = get_model()
    num_epochs = 1
    batch_size = 4
    ctx = mx.cpu()
    loss = gluon.loss.L2Loss()
    acc = mx.metric.Accuracy()
    net.initialize(ctx=ctx)
    trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.001})
    est = estimator.Estimator(net=net,
                              loss=loss,
                              metrics=acc,
                              trainers=trainer,
                              context=ctx)
    in_data = mx.nd.random.uniform(shape=(10, 3))
    out_data = mx.nd.random.uniform(shape=(10, 4))
    # Input dataloader
    dataset = gluon.data.dataset.ArrayDataset(in_data, out_data)
    train_dataloader = gluon.data.DataLoader(dataset, batch_size=batch_size)
    est.fit(train_data=train_dataloader,
            epochs=num_epochs,
            batch_size=batch_size)

    # Input dataiter
    train_dataiter = mx.io.NDArrayIter(data=in_data, label=out_data, batch_size=batch_size)
    est.fit(train_data=train_dataiter,
            epochs=num_epochs,
            batch_size=batch_size)

    # Input NDArray
    with assert_raises(ValueError):
        est.fit(train_data=[in_data, out_data],
                epochs=num_epochs,
                batch_size=batch_size)


@with_seed()
def test_validation():
    net = get_model()
    num_epochs = 1
    batch_size = 4
    ctx = mx.cpu()
    loss = gluon.loss.L2Loss()
    acc = mx.metric.Accuracy()
    net.initialize(ctx=ctx)
    trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.001})
    est = estimator.Estimator(net=net,
                              loss=loss,
                              metrics=acc,
                              trainers=trainer,
                              context=ctx)
    in_data = mx.nd.random.uniform(shape=(10, 3))
    out_data = mx.nd.random.uniform(shape=(10, 4))
    # Input dataloader
    dataset = gluon.data.dataset.ArrayDataset(in_data, out_data)
    train_dataloader = gluon.data.DataLoader(dataset, batch_size=batch_size)
    val_dataloader = gluon.data.DataLoader(dataset, batch_size=batch_size)
    est.fit(train_data=train_dataloader,
            val_data=val_dataloader,
            epochs=num_epochs,
            batch_size=batch_size)

    # Input dataiter
    train_dataiter = mx.io.NDArrayIter(data=in_data, label=out_data, batch_size=batch_size)
    val_dataiter = mx.io.NDArrayIter(data=in_data, label=out_data, batch_size=batch_size)
    est.fit(train_data=train_dataiter,
            val_data=val_dataiter,
            epochs=num_epochs,
            batch_size=batch_size)
    # Input NDArray
    with assert_raises(ValueError):
        est.fit(train_data=[in_data, out_data],
                val_data=[in_data, out_data],
                epochs=num_epochs,
                batch_size=batch_size)


@with_seed()
def test_initializer():
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
    # catch no init and no trainer warning
    with warnings.catch_warnings(record=True) as w:
        est = estimator.Estimator(net=net,
                                  loss=loss,
                                  metrics=acc,
                                  context=ctx)
        assert len(w) == 2
    est.fit(train_data=train_data,
            epochs=num_epochs,
            batch_size=batch_size)

    # different initializer for net and estimator
    net = get_model()
    net.initialize(mx.init.Xavier(), ctx=ctx)
    trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.001})
    # catch no trainer and reinit warning
    with warnings.catch_warnings(record=True) as w:
        est = estimator.Estimator(net=net,
                                  loss=loss,
                                  metrics=acc,
                                  initializer=mx.init.MSRAPrelu(),
                                  trainers=trainer,
                                  context=ctx)
        assert len(w) == 2
    est.fit(train_data=train_data,
            epochs=num_epochs,
            batch_size=batch_size)
    # input invalid initializer
    net = get_model()
    with assert_raises(ValueError):
        est = estimator.Estimator(net=net,
                                  loss=loss,
                                  metrics=acc,
                                  initializer='xavier',
                                  trainers=trainer,
                                  context=ctx)

@with_seed()
def test_trainer():
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
        est = estimator.Estimator(net=net,
                                  loss=loss,
                                  metrics=acc,
                                  context=ctx)
        assert len(w) == 1
    est.fit(train_data=train_data,
            epochs=num_epochs,
            batch_size=batch_size)

    # input invalid trainer
    trainer = 'sgd'
    with assert_raises(ValueError):
        est = estimator.Estimator(net=net,
                                  loss=loss,
                                  metrics=acc,
                                  trainers=trainer,
                                  context=ctx)

@with_seed
def test_metric():
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
    est = estimator.Estimator(net=net,
                              loss=loss,
                              trainers=trainer,
                              context=ctx)
    est.fit(train_data=train_data,
            epochs=num_epochs,
            batch_size=batch_size)
    # input list of metrics
    metrics = [mx.metric.Accuracy(), mx.metric.Accuracy()]
    est = estimator.Estimator(net=net,
                              loss=loss,
                              metrics=metrics,
                              trainers=trainer,
                              context=ctx)
    est.fit(train_data=train_data,
            epochs=num_epochs,
            batch_size=batch_size)
    # input invalid metric
    with assert_raises(ValueError):
        est = estimator.Estimator(net=net,
                                  loss=loss,
                                  metrics='acc',
                                  trainers=trainer,
                                  context=ctx)

@with_seed
def test_loss():
    net = get_model()
    num_epochs = 1
    batch_size = 4
    ctx = mx.cpu()
    in_data = mx.nd.random.uniform(shape=(10, 3))
    out_data = mx.nd.random.uniform(shape=(10, 4))
    dataset = gluon.data.dataset.ArrayDataset(in_data, out_data)
    train_data = gluon.data.DataLoader(dataset, batch_size=batch_size)
    acc = mx.metric.Accuracy()
    net.initialize(ctx=ctx)
    trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.001})
    # input no loss, catch warning
    with warnings.catch_warnings(record=True) as w:
        est = estimator.Estimator(net=net,
                                  trainers=trainer,
                                  metrics=acc,
                                  context=ctx)
        assert len(w) == 1
    est.fit(train_data=train_data,
            epochs=num_epochs,
            batch_size=batch_size)
    # input list of losses
    loss = [gluon.loss.L2Loss(), gluon.loss.L2Loss()]
    est = estimator.Estimator(net=net,
                              loss=loss,
                              metrics=acc,
                              trainers=trainer,
                              context=ctx)
    est.fit(train_data=train_data,
            epochs=num_epochs,
            batch_size=batch_size)
    # input invalid loss
    with assert_raises(ValueError):
        est = estimator.Estimator(net=net,
                                  loss='mse',
                                  metrics=acc,
                                  trainers=trainer,
                                  context=ctx)

@with_seed()
def test_batch_size():
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
    trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.001})
    est = estimator.Estimator(net=net,
                              loss=loss,
                              metrics=acc,
                              trainers=trainer,
                              context=ctx)
    # check auto infer batch size
    est.fit(train_data=train_data,
            epochs=num_epochs)

    # check invalid batch size
    with assert_raises(ValueError):
        est.fit(train_data=train_data,
                epochs=num_epochs,
                batch_size=batch_size+1)

@with_seed()
def test_context():
    # GPU needed??
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
    trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.001})
    # input no context
    est = estimator.Estimator(net=net,
                              loss=loss,
                              metrics=acc,
                              trainers=trainer)
    est.fit(train_data=train_data,
            epochs=num_epochs,
            batch_size=batch_size)
    # input list of context

    # input invalid context

if __name__ == '__main__':
    import nose
    nose.runmodule()