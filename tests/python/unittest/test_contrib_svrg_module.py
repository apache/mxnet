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
import numpy as np
from common import with_seed, assertRaises
from mxnet.contrib.svrg_optimization.svrg_module import SVRGModule
from mxnet.test_utils import *
import unittest

def setup():
    train_data = np.random.randint(1, 5, [1000, 2])
    weights = np.array([1.0, 2.0])
    train_label = train_data.dot(weights)

    di = mx.io.NDArrayIter(train_data, train_label, batch_size=32, shuffle=True, label_name='lin_reg_label')
    X = mx.sym.Variable('data')
    Y = mx.symbol.Variable('lin_reg_label')
    fully_connected_layer = mx.sym.FullyConnected(data=X, name='fc1', num_hidden=1)
    lro = mx.sym.LinearRegressionOutput(data=fully_connected_layer, label=Y, name="lro")

    mod = SVRGModule(
        symbol=lro,
        data_names=['data'],
        label_names=['lin_reg_label'], update_freq=2)
    mod.bind(data_shapes=di.provide_data, label_shapes=di.provide_label)
    mod.init_params(initializer=mx.init.Uniform(0.01), allow_missing=False, force_init=False, allow_extra=False)

    return di, mod


def test_bind_module():
    _, mod = setup()
    assert mod.binded == True
    assert mod._mod_aux.binded == True


def test_module_init():
    _, mod = setup()
    assert mod._mod_aux is not None


def test_module_initializer():
    def regression_model(m):
        x = mx.symbol.var("data", stype='csr')
        v = mx.symbol.var("v", shape=(m, 1), init=mx.init.Uniform(scale=.1),
                          stype='row_sparse')
        model = mx.symbol.dot(lhs=x, rhs=v)
        y = mx.symbol.Variable("label")
        model = mx.symbol.LinearRegressionOutput(data=model, label=y, name="out")
        return model

    #shape of the data
    n, m = 128, 100
    model = regression_model(m)

    data = mx.nd.zeros(shape=(n, m), stype='csr')
    label = mx.nd.zeros((n, 1))
    iterator = mx.io.NDArrayIter(data=data, label={'label': label},
                                 batch_size=n, last_batch_handle='discard')

    # create module
    mod = SVRGModule(symbol=model, data_names=['data'], label_names=['label'], update_freq=2)
    mod.bind(data_shapes=iterator.provide_data, label_shapes=iterator.provide_label)
    mod.init_params()
    v = mod._arg_params['v']
    assert v.stype == 'row_sparse'
    assert np.sum(v.asnumpy()) != 0


def test_module_bind():
    x = mx.sym.Variable("data")
    net = mx.sym.FullyConnected(x, num_hidden=1)

    mod = SVRGModule(symbol=net, data_names=['data'], label_names=None, update_freq=2)
    assertRaises(TypeError, mod.bind, data_shapes=['data', mx.nd.zeros(shape=(2, 1))])

    mod.bind(data_shapes=[('data', (2, 1))])
    assert mod.binded == True
    assert mod._mod_aux.binded == True


@unittest.skip("Flaky test https://gitsvrhub.com/apache/incubator-mxnet/issues/12510")
@with_seed()
def test_module_save_load():
    import tempfile
    import os

    x = mx.sym.Variable("data")
    y = mx.sym.Variable("softmax_label")
    net = mx.sym.FullyConnected(x, y, num_hidden=1)

    mod = SVRGModule(symbol=net, data_names=['data'], label_names=['softmax_label'], update_freq=2)
    mod.bind(data_shapes=[('data', (1, 1))])
    mod.init_params()
    mod.init_optimizer(optimizer='sgd', optimizer_params={'learning_rate': 0.1})
    mod.update()

    # Create tempfile
    tmp = tempfile.mkdtemp()
    tmp_file = os.path.join(tmp, 'svrg_test_output')
    mod.save_checkpoint(tmp_file, 0, save_optimizer_states=True)

    mod2 = SVRGModule.load(tmp_file, 0, load_optimizer_states=True, data_names=('data', ))
    mod2.bind(data_shapes=[('data', (1, 1))])
    mod2.init_optimizer(optimizer_params={'learning_rate': 0.1})
    assert mod._symbol.tojson() == mod2._symbol.tojson()

    # Multi-device
    mod3 = SVRGModule(symbol=net, data_names=['data'], label_names=['softmax_label'], update_freq=3,
                     context=[mx.cpu(0), mx.cpu(1)])
    mod3.bind(data_shapes=[('data', (10, 10))])
    mod3.init_params()
    mod3.init_optimizer(optimizer_params={'learning_rate': 1.0})
    mod3.update()
    mod3.save_checkpoint(tmp_file, 0, save_optimizer_states=True)

    mod4 = SVRGModule.load(tmp_file, 0, load_optimizer_states=True, data_names=('data', ))
    mod4.bind(data_shapes=[('data', (10, 10))])
    mod4.init_optimizer(optimizer_params={'learning_rate': 1.0})
    assert mod3._symbol.tojson() == mod4._symbol.tojson()


@unittest.skip("Flaky test https://github.com/apache/incubator-mxnet/issues/12510")
@with_seed()
def test_svrgmodule_reshape():
    data = mx.sym.Variable("data")
    sym = mx.sym.FullyConnected(data=data, num_hidden=4, name='fc')

    dshape=(3, 4)
    mod = SVRGModule(sym, data_names=["data"], label_names=None, context=[mx.cpu(0), mx.cpu(1)], update_freq=2)
    mod.bind(data_shapes=[('data', dshape)])
    mod.init_params()
    mod._mod_aux.init_params()
    mod.init_optimizer(optimizer_params={"learning_rate": 1.0})

    data_batch = mx.io.DataBatch(data=[mx.nd.ones(dshape)], label=None)
    mod.forward(data_batch)
    mod.backward([mx.nd.ones(dshape)])
    mod.update()
    assert mod.get_outputs()[0].shape == dshape

    dshape = (2, 4)
    mod.reshape(data_shapes=[('data', dshape)])
    mod.forward(mx.io.DataBatch(data=[mx.nd.ones(dshape)],
                                label=None))
    mod.backward([mx.nd.ones(dshape)])
    mod.update()
    assert mod.get_outputs()[0].shape == dshape


@unittest.skip("Flaky test https://github.com/apache/incubator-mxnet/issues/12510")
@with_seed()
def test_update_full_grad():
    def create_network():
        train_data = np.random.randint(1, 5, [10, 2])
        weights = np.array([1.0, 2.0])
        train_label = train_data.dot(weights)

        di = mx.io.NDArrayIter(train_data, train_label, batch_size=5, shuffle=True, label_name='lin_reg_label')
        X = mx.sym.Variable('data')
        Y = mx.symbol.Variable('lin_reg_label')
        fully_connected_layer = mx.sym.FullyConnected(data=X, name='fc1', num_hidden=1)
        lro = mx.sym.LinearRegressionOutput(data=fully_connected_layer, label=Y, name="lro")

        mod = SVRGModule(
            symbol=lro,
            data_names=['data'],
            label_names=['lin_reg_label'], update_freq=2)
        mod.bind(data_shapes=di.provide_data, label_shapes=di.provide_label)
        mod.init_params(initializer=mx.init.One(), allow_missing=False, force_init=False, allow_extra=False)
        mod.init_optimizer(kvstore='local', optimizer='sgd', optimizer_params=(('learning_rate', 0.01),),
                           force_init=False)
        return di, mod

    di, svrg_mod = create_network()

    # Calculates the average of full gradients over number batches
    full_grads_weights = mx.nd.zeros(shape=svrg_mod.get_params()[0]['fc1_weight'].shape)
    arg, aux = svrg_mod.get_params()
    svrg_mod._mod_aux.set_params(arg_params=arg, aux_params=aux)
    num_batch = 2

    for batch in di:
        svrg_mod.forward(batch)
        svrg_mod.backward()
        full_grads_weights = mx.nd.broadcast_add(svrg_mod._exec_group.grad_arrays[0][0], full_grads_weights, axis=0)
    full_grads_weights /= num_batch

    di.reset()
    svrg_mod.update_full_grads(di)
    assert same(full_grads_weights, svrg_mod._param_dict[0]['fc1_weight'])


@unittest.skip("Flaky test https://github.com/apache/incubator-mxnet/issues/12510")
@with_seed()
def test_svrg_with_sgd():
    def create_module_with_sgd():
        train_data = np.random.randint(1, 5, [100, 2])
        weights = np.array([1.0, 2.0])
        train_label = train_data.dot(weights)

        di = mx.io.NDArrayIter(train_data, train_label, batch_size=10, shuffle=True, label_name='lin_reg_label')
        X = mx.sym.Variable('data')
        Y = mx.symbol.Variable('lin_reg_label')
        fully_connected_layer = mx.sym.FullyConnected(data=X, name='fc1', num_hidden=1)
        lro = mx.sym.LinearRegressionOutput(data=fully_connected_layer, label=Y, name="lro")

        reg_mod = mx.mod.Module(
            symbol=lro,
            data_names=['data'],
            label_names=['lin_reg_label'])
        reg_mod.bind(data_shapes=di.provide_data, label_shapes=di.provide_label)
        reg_mod.init_params(initializer=mx.init.One(), allow_missing=False, force_init=False, allow_extra=False)
        reg_mod.init_optimizer(kvstore='local', optimizer='sgd', optimizer_params=(('learning_rate', 0.01),))

        svrg_mod = SVRGModule(symbol=lro,
            data_names=['data'],
            label_names=['lin_reg_label'],
            update_freq=2)
        svrg_mod.bind(data_shapes=di.provide_data, label_shapes=di.provide_label)
        svrg_mod.init_params(initializer=mx.init.One(), allow_missing=False, force_init=False, allow_extra=False)
        svrg_mod.init_optimizer(kvstore='local', optimizer='sgd', optimizer_params=(('learning_rate', 0.01),))

        return di,reg_mod, svrg_mod

    di, reg_mod, svrg_mod = create_module_with_sgd()
    num_epoch = 10

    # Use metric MSE
    metrics = mx.metric.create("mse")

    # Train with SVRGModule
    for e in range(num_epoch):
        metrics.reset()
        if e % svrg_mod.update_freq == 0:
            svrg_mod.update_full_grads(di)
        di.reset()
        for batch in di:
            svrg_mod.forward_backward(data_batch=batch)
            svrg_mod.update()
            svrg_mod.update_metric(metrics, batch.label)
    svrg_mse = metrics.get()[1]

    # Train with SGD standard Module
    di.reset()
    for e in range(num_epoch):
        metrics.reset()
        di.reset()
        for batch in di:
            reg_mod.forward_backward(data_batch=batch)
            reg_mod.update()
            reg_mod.update_metric(metrics, batch.label)
    sgd_mse = metrics.get()[1]

    assert svrg_mse < sgd_mse


@unittest.skip("Flaky test https://github.com/apache/incubator-mxnet/issues/12510")
@with_seed()
def test_accumulate_kvstore():
    # Test KVStore behavior when push a list of values
    kv = mx.kv.create('local')
    kv.init("fc1_weight", mx.nd.zeros(shape=(1, 2)))
    kv.init("fc1_weight_full", mx.nd.zeros(shape=(1, 2)))
    b = [mx.nd.ones(shape=(1, 2)) for i in range(4)]
    a = mx.nd.zeros(shape=(1, 2))
    kv.push("fc1_weight_full", b)
    kv.pull("fc1_weight_full", out=a)
    assert same(a, [mx.nd.array([4, 4])])
    assert kv.num_workers == 1

    # Test accumulate in KVStore and allocate gradients
    kv_test = mx.kv.create('local')
    _, svrg_mod = setup()
    svrg_mod.init_optimizer(kvstore=kv_test, optimizer='sgd', optimizer_params=(('learning_rate', 0.01),),
                            force_init=False)
    svrg_mod._accumulate_kvstore("fc1_weight", b)
    assert len(svrg_mod._param_dict) == svrg_mod._ctx_len
    assert same(svrg_mod._param_dict[0]["fc1_weight"], b[0])


@unittest.skip("Flaky test https://github.com/apache/incubator-mxnet/issues/12510")
@with_seed()
def test_fit():
    di, mod = setup()
    num_epoch = 100
    metric = mx.metric.create("mse")
    mod.fit(di, eval_metric=metric, optimizer='sgd', optimizer_params=(('learning_rate', 0.025),), num_epoch=num_epoch,
            kvstore='local')

    # Estimated MSE for using SGD optimizer of lr = 0.025, SVRG MSE should be smaller
    estimated_mse = 1e-5
    assert metric.get()[1] < estimated_mse


if __name__ == "__main__":
    import nose
    nose.runmodule()
