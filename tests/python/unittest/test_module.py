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
import mxnet.ndarray as nd
from mxnet.test_utils import *
import numpy as np
from functools import reduce
from mxnet.module.executor_group import DataParallelExecutorGroup
from common import setup_module, with_seed, assertRaises, teardown_module
from collections import namedtuple
curr_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
sys.path.insert(0, os.path.join(curr_path, "../train"))
import pytest


@with_seed()
def test_module_dtype():
    dtype = np.float16
    dshape = (3, 8, 7)

    sym = mx.sym.Variable('data')
    sym = mx.sym.Activation(data=sym, act_type='relu', __layout__='TNC')

    mod = mx.mod.Module(sym, ('data',), None, context=[mx.cpu(0), mx.cpu(1)])
    mod.bind(data_shapes=[mx.io.DataDesc('data', dshape, dtype, layout='TNC')])
    mod.init_params()
    mod.forward(mx.io.DataBatch(data=[mx.nd.ones(dshape, dtype=dtype)],
                              label=None))
    mod.backward([mx.nd.ones(dshape, dtype=dtype)])

    for x in mod.get_outputs():
      assert x.dtype == dtype


def test_module_bind():
    sym = mx.sym.Variable('data')
    sym = mx.sym.Activation(data=sym, act_type='relu', __layout__='TNC')

    mod = mx.mod.Module(sym, ('data',), None, context=[mx.cpu(0), mx.cpu(1)])
    assertRaises(TypeError, mod.bind, data_shapes=[('data', mx.nd.array([10,10]))])
    assert mod.binded == False

    mod.bind(data_shapes=[('data', (10,10))])
    assert mod.binded == True


@with_seed()
def test_module_input_grads():
    a = mx.sym.Variable('a', __layout__='NC')
    b = mx.sym.Variable('b', __layout__='NC')
    c = mx.sym.Variable('c', __layout__='NC')

    c = a + 2 * b + 3 * c
    net = mx.mod.Module(c, data_names=['b', 'c', 'a'], label_names=None,
                        context=[mx.cpu(0), mx.cpu(1)])
    net.bind(data_shapes=[['b', (5, 5)], ['c', (5, 5)], ['a', (5, 5)]],
             label_shapes=None, inputs_need_grad=True)
    net.init_params()

    net.forward(data_batch=mx.io.DataBatch(data=[nd.ones((5, 5)),
                                                 nd.ones((5, 5)),
                                                 nd.ones((5, 5))]))
    net.backward(out_grads=[nd.ones((5, 5))])
    input_grads = net.get_input_grads()
    b_grad = input_grads[0].asnumpy()
    c_grad = input_grads[1].asnumpy()
    a_grad = input_grads[2].asnumpy()
    assert np.all(a_grad == 1), a_grad
    assert np.all(b_grad == 2), b_grad
    assert np.all(c_grad == 3), c_grad


@with_seed()
def test_module_ctx_group():
    def check_module_ctx_group(ctxs, group2ctxs, grad_ctxs=None):
        with mx.AttrScope(ctx_group='dev1'):
            a = mx.symbol.Variable('a')
            a = a * 2
        with mx.AttrScope(ctx_group='dev2'):
            b = mx.symbol.Variable('b')
            c = a + b
        shape = (2, 5)
        mod1 = mx.mod.Module(c, context=ctxs, data_names=['a', 'b'], label_names=None,
                             group2ctxs=group2ctxs)
        mod1.bind(data_shapes=[['a', shape], ['b', shape]], inputs_need_grad=True)
        mod1.init_params()
        mod1.forward(data_batch=mx.io.DataBatch(data=[mx.nd.ones(shape), mx.nd.ones(shape)]), is_train=True)
        mod1.backward([mx.nd.ones(shape)])
        mod1_input_grads = mod1.get_input_grads()

        mod2 = mx.mod.Module(c, context=ctxs, data_names=['a', 'b'], label_names=None)
        mod2.bind(data_shapes=[['a', shape], ['b', shape]], inputs_need_grad=True)
        mod2.init_params()
        mod2.forward(data_batch=mx.io.DataBatch(data=[mx.nd.ones(shape), mx.nd.ones(shape)]), is_train=True)
        mod2.backward([mx.nd.ones(shape)])
        mod2_input_grads = mod2.get_input_grads()

        if grad_ctxs is not None:
            assert(mod1_input_grads[0].context == grad_ctxs[0])
            assert(mod1_input_grads[1].context == grad_ctxs[1])
        assert(np.all(mod1_input_grads[0].asnumpy() == mod2_input_grads[0].asnumpy()))
        assert(np.all(mod1_input_grads[1].asnumpy() == mod2_input_grads[1].asnumpy()))

    check_module_ctx_group([mx.cpu(0)], {'dev1': mx.cpu(1), 'dev2': mx.cpu(2)}, grad_ctxs=[mx.cpu(1), mx.cpu(2)])
    check_module_ctx_group([mx.cpu(0), mx.cpu(1)],
        [{'dev1': mx.cpu(2), 'dev2': mx.cpu(3)}, {'dev1': mx.cpu(4), 'dev2': mx.cpu(5)}])
    check_module_ctx_group([mx.cpu(0), mx.cpu(1)], {'dev1': mx.cpu(2), 'dev2': mx.cpu(3)})
    check_module_ctx_group([mx.cpu(0), mx.cpu(1)], {'dev1': mx.cpu(2), 'dev2': [mx.cpu(3)]})
    check_module_ctx_group([mx.cpu(0), mx.cpu(1)], {'dev1':mx.cpu(2), 'dev2':[mx.cpu(3), mx.cpu(3)]})
    check_module_ctx_group([mx.cpu(0), mx.cpu(1)],
        {'dev1':[mx.cpu(2), mx.cpu(2)], 'dev2':[mx.cpu(3), mx.cpu(3)]})

@with_seed()
def test_bucket_module_ctx_group():
    num_hidden = 10
    batch_size = 5
    def sym_gen(seq_len):
        with mx.AttrScope(ctx_group='dev1'):
            data = mx.symbol.Variable('data')
            weight = mx.symbol.Variable('dev1_weight')
            bias = mx.symbol.Variable('dev1_bias')
            fc = data
            for i in range(seq_len):
                fc  = mx.symbol.FullyConnected(data=fc, weight=weight, bias=bias,
                                               name='dev1_fc_%d' % i, num_hidden=num_hidden)
        with mx.AttrScope(ctx_group='dev2'):
            label = mx.symbol.Variable('label')
            weight = mx.symbol.Variable('dev2_weight')
            bias = mx.symbol.Variable('dev2_bias')
            for i in range(seq_len):
                fc  = mx.symbol.FullyConnected(data=fc, weight=weight, bias=bias,
                                               name='dev2_fc_%d' % i, num_hidden=num_hidden)
            sym = mx.symbol.SoftmaxOutput(fc, label, name='softmax')

        return sym, ('data',), ('label',)

    mod = mx.mod.BucketingModule(sym_gen=sym_gen, default_bucket_key=10, context=[mx.cpu(0)],
                                 group2ctxs=[{'dev1': mx.cpu(1), 'dev2': mx.cpu(2)}])
    mod.bind(data_shapes=[['data', (batch_size, num_hidden)]],
             label_shapes=[['label', (batch_size,)]],
             for_training=True, inputs_need_grad=True)
    assert(mod.binded)

@with_seed()
def test_module_layout():
    sym = mx.sym.Variable('data')
    sym = mx.sym.Activation(data=sym, act_type='relu', __layout__='TNC')

    dshape = (3, 8, 7)
    mod = mx.mod.Module(sym, ('data',), None, context=[mx.cpu(0), mx.cpu(1)])
    mod.bind(data_shapes=[mx.io.DataDesc('data', dshape, layout='TNC')])
    mod.init_params()
    mod.forward(mx.io.DataBatch(data=[mx.nd.ones(dshape)],
                                label=None))
    mod.backward([mx.nd.ones(dshape)])
    assert mod.get_outputs()[0].shape == dshape

    hdshape = (3, 4, 7)
    for x in mod.get_outputs(merge_multi_context=False)[0]:
        assert x.shape == hdshape


@with_seed()
@pytest.mark.parametrize('ctx,get_updater', [
    (mx.cpu(), lambda m: m._updater),
    ([mx.cpu(0), mx.cpu(1)], lambda m: m._kvstore._updater)
])
def test_save_load(ctx, get_updater, tmpdir):
    previous_update_on_kvstore = os.getenv('MXNET_UPDATE_ON_KVSTORE', "1")
    os.putenv('MXNET_UPDATE_ON_KVSTORE', '1')
    def dict_equ(a, b):
        assert set(a) == set(b)
        for k in a:
            assert (a[k].asnumpy() == b[k].asnumpy()).all()

    sym = mx.sym.Variable('data')
    sym = mx.sym.FullyConnected(sym, num_hidden=100)

    path = str(tmpdir.join('test'))
    mod = mx.mod.Module(sym, ('data',), context=ctx)
    mod.bind(data_shapes=[('data', (10, 10))])
    mod.init_params()
    mod.init_optimizer(optimizer_params={'learning_rate':0.1, 'momentum':0.9})
    mod.update()
    mod.save_checkpoint(path, 0, save_optimizer_states=True)

    mod2 = mx.mod.Module.load(path, 0, load_optimizer_states=True, data_names=('data',))
    mod2.bind(data_shapes=[('data', (10, 10))])
    mod2.init_optimizer(optimizer_params={'learning_rate':0.1, 'momentum':0.9})
    assert mod._symbol.tojson() == mod2._symbol.tojson()
    dict_equ(mod.get_params()[0], mod2.get_params()[0])
    dict_equ(get_updater(mod).states, mod2._updater.states)

    os.putenv('MXNET_UPDATE_ON_KVSTORE', previous_update_on_kvstore)


@with_seed()
def test_module_reshape():
    data = mx.sym.Variable('data')
    sym = mx.sym.FullyConnected(data, num_hidden=20, name='fc')

    dshape = (7, 20)
    mod = mx.mod.Module(sym, ('data',), None, context=[mx.cpu(0), mx.cpu(1)])
    mod.bind(data_shapes=[('data', dshape)])
    mod.init_params()
    mod.init_optimizer(optimizer_params={'learning_rate': 1})

    mod.forward(mx.io.DataBatch(data=[mx.nd.ones(dshape)],
                                label=None))
    mod.backward([mx.nd.ones(dshape)])
    mod.update()
    assert mod.get_outputs()[0].shape == dshape
    assert (mod.get_params()[0]['fc_bias'].asnumpy() == -1).all()

    dshape = (14, 20)
    mod.reshape(data_shapes=[('data', dshape)])
    mod.forward(mx.io.DataBatch(data=[mx.nd.ones(dshape)],
                                label=None))
    mod.backward([mx.nd.ones(dshape)])
    mod.update()
    assert mod.get_outputs()[0].shape == dshape
    assert (mod.get_params()[0]['fc_bias'].asnumpy() == -3).all()



# roywei: Getting rid of fixed seed as flakiness could not be reproduced,
# tracked at: https://github.com/apache/incubator-mxnet/issues/11705
@with_seed()
def test_module_set_params():
    # data iter
    data = mx.nd.array([[0.05, .10]]);
    label = mx.nd.array([[.01, 0.99]]);
    train_data = mx.io.NDArrayIter(data, label, batch_size=1)

    # symbols
    x = mx.symbol.Variable('data')
    x = mx.symbol.FullyConnected(name='fc_0', data=x, num_hidden=2)
    x = mx.symbol.Activation(name="act_0", data=x, act_type='sigmoid')
    x = mx.symbol.FullyConnected(name='fc_1', data=x, num_hidden=2)
    x = mx.symbol.Activation(name="act_1", data=x, act_type='sigmoid')
    x = mx.symbol.LinearRegressionOutput(data=x, name='softmax', grad_scale=2)

    # create module
    mod = mx.mod.Module(x, context=[mx.cpu()]);
    mod.bind(train_data.provide_data, label_shapes=train_data.provide_label,
             for_training=True)

    arg_params_correct = {'fc_0_weight': mx.nd.array([[.15, .20], [.25, .30]]),
                  'fc_0_bias'  : mx.nd.array([.35, .35]),
                  'fc_1_weight': mx.nd.array([[.40, .45], [.50, .55]]),
                  'fc_1_bias'  : mx.nd.array([.60, .60])}

    arg_params_missing = {'fc_0_weight': mx.nd.array([[.15, .20], [.25, .30]]),
                  'fc_0_bias'  : mx.nd.array([.35, .35]),
                  'fc_1_weight': mx.nd.array([[.40, .45], [.50, .55]])}

    arg_params_extra = {'fc_0_weight': mx.nd.array([[.15, .20], [.25, .30]]),
                  'fc_0_bias'  : mx.nd.array([.35, .35]),
                  'fc_1_weight': mx.nd.array([[.40, .45], [.50, .55]]),
                  'fc_1_bias'  : mx.nd.array([.60, .60]),
                  'fc_2_weight': mx.nd.array([.60, .60])}

    arg_params_missing_extra = {'fc_2_weight': mx.nd.array([.60, .60])}

    # test regular set_params
    mod.set_params(force_init=True, arg_params=arg_params_correct, aux_params={})

    # test allow missing
    mod.set_params(force_init=True, arg_params=arg_params_missing, aux_params={}, allow_missing=True)
    assertRaises(RuntimeError, mod.set_params,
                 force_init=True, arg_params=arg_params_missing,
                 aux_params={}, allow_missing=False)

    # test allow extra
    mod.set_params(force_init=True, arg_params=arg_params_extra, aux_params={}, allow_missing=True, allow_extra=True)
    assertRaises(ValueError, mod.set_params,
                 force_init=True, arg_params=arg_params_extra,
                 aux_params={}, allow_missing=True, allow_extra=False)

    # test allow missing + extra,
    assertRaises(RuntimeError, mod.set_params,
                 force_init=True, arg_params=arg_params_missing_extra,
                 aux_params={}, allow_missing=False, allow_extra=False)

    # test allow missing + extra, this will throw a runtime error
    assertRaises(ValueError, mod.set_params,
                 force_init=True, arg_params=arg_params_missing_extra,
                 aux_params={}, allow_missing=True, allow_extra=False)


@with_seed()
@pytest.mark.garbage_expected
def test_monitor():
    # data iter
    data = mx.nd.array([[0.05, .10]]);
    label = mx.nd.array([[.01, 0.99]]);
    train_data = mx.io.NDArrayIter(data, label, batch_size=1)

    # symbols
    x = mx.symbol.Variable('data')
    x = mx.symbol.FullyConnected(name='fc_0', data=x, num_hidden=2)
    x = mx.symbol.Activation(name="act_0", data=x, act_type='sigmoid')
    x = mx.symbol.FullyConnected(name='fc_1', data=x, num_hidden=2)
    x = mx.symbol.Activation(name="act_1", data=x, act_type='sigmoid')
    x = mx.symbol.LinearRegressionOutput(data=x, name='softmax', grad_scale=2)

    # create monitor
    def mean_abs(x):
        sum_abs = mx.ndarray.sum(mx.ndarray.abs(x))
        return mx.ndarray.divide(sum_abs, reduce(lambda x, y: x * y, x.shape))
    mon = mx.mon.Monitor(1, stat_func=mean_abs, pattern='.*', sort=True)

    # create module
    mod = mx.mod.Module(x, context=[mx.cpu()]);
    mod.bind(train_data.provide_data, label_shapes=train_data.provide_label,
                    for_training=True)
    mod.install_monitor(mon)
    arg_params = {'fc_0_weight': mx.nd.array([[.15, .20], [.25, .30]]),
                  'fc_0_bias'  : mx.nd.array([.35, .35]),
                  'fc_1_weight': mx.nd.array([[.40, .45], [.50, .55]]),
                  'fc_1_bias'  : mx.nd.array([.60, .60])}
    mod.init_params(arg_params=arg_params)

    data_iter = iter(train_data)
    data_batch = next(data_iter)
    mon.tic()
    mod.forward_backward(data_batch)
    res = mon.toc()
    keys = ['act_0', 'act_1', 'data', 'fc_0', 'fc_1', 'softmax']
    mon_result_counts = [0, 0, 0, 0, 0, 0]
    assert(len(res) == 21)
    for n, k, v in res:
        for idx, key in enumerate(keys):
            if k.startswith(key):
                mon_result_counts[idx] += 1
                break
    assert(mon_result_counts == [2, 2, 1, 6, 6, 4])


@with_seed()
def test_factorization_machine_module():
    """ Test factorization machine model with sparse operators """
    # this unit test is to test the flow, training accuracy is tested in another test
    def check_factorization_machine_module(num_epochs=None):
        print("check_factorization_machine_module")

        def fm(factor_size, feature_dim, init):
            x = mx.symbol.Variable("data", stype='csr')
            v = mx.symbol.Variable("v", shape=(feature_dim, factor_size),
                                   init=init, stype='row_sparse')

            w1_weight = mx.symbol.var('w1_weight', shape=(feature_dim, 1),
                                      init=init, stype='row_sparse')
            w1_bias = mx.symbol.var('w1_bias', shape=(1))
            w1 = mx.symbol.broadcast_add(mx.symbol.dot(x, w1_weight), w1_bias)

            v_s = mx.symbol._internal._square_sum(data=v, axis=1, keepdims=True)
            x_s = mx.symbol.square(data=x)
            bd_sum = mx.sym.dot(x_s, v_s)

            w2 = mx.symbol.dot(x, v)
            w2_squared = 0.5 * mx.symbol.square(data=w2)

            w_all = mx.symbol.Concat(w1, w2_squared, dim=1)
            sum1 = mx.symbol.sum(data=w_all, axis=1, keepdims=True)
            sum2 = 0.5 * mx.symbol.negative(bd_sum)
            model = mx.sym.elemwise_add(sum1, sum2)

            y = mx.symbol.Variable("label")
            model = mx.symbol.LinearRegressionOutput(data=model, label=y)
            return model

        # model
        init = mx.initializer.Normal(sigma=0.01)
        factor_size = 4
        feature_dim = 10000
        model = fm(factor_size, feature_dim, init)

        # data iter
        num_batches = 5
        batch_size = 64
        num_samples = batch_size * num_batches
        # generate some random csr data
        csr_nd = rand_ndarray((num_samples, feature_dim), 'csr', 0.1)
        label = mx.nd.ones((num_samples,1))
        # the alternative is to use LibSVMIter
        train_iter = mx.io.NDArrayIter(data=csr_nd,
                                       label={'label':label},
                                       batch_size=batch_size,
                                       last_batch_handle='discard')
        # create module
        mod = mx.mod.Module(symbol=model, data_names=['data'], label_names=['label'])
        # allocate memory by given the input data and lable shapes
        mod.bind(data_shapes=train_iter.provide_data, label_shapes=train_iter.provide_label)
        # initialize parameters by uniform random numbers
        mod.init_params(initializer=init)

        # use Sparse SGD with learning rate 0.1 to train
        sgd = mx.optimizer.SGD(momentum=0.1, clip_gradient=5.0, learning_rate=0.01,
                               rescale_grad=1.0/batch_size)
        mod.init_optimizer(optimizer=sgd)
        if num_epochs is None:
            num_epochs = 50
        expected_accuracy = 0.02

	# use accuracy as the metric
        metric = mx.gluon.metric.create('MSE')
        # train 'num_epochs' epoch
        for epoch in range(num_epochs):
            train_iter.reset()
            metric.reset()
            for batch in train_iter:
                mod.forward(batch, is_train=True)       # compute predictions
                mod.update_metric(metric, batch.label)  # accumulate prediction accuracy
                mod.backward()                          # compute gradients
                mod.update()                            # update parameters
            print('Epoch %d, Training %s' % (epoch, metric.get()))
        if num_epochs > 1:
            assert(metric.get()[1] < expected_accuracy)

    check_factorization_machine_module()

@with_seed()
def test_module_initializer():
    def regression_model(m):
         x = mx.symbol.var("data", stype='csr')
         v = mx.symbol.var("v", shape=(m, 1), init=mx.init.Uniform(scale=.1),
                                stype='row_sparse')
         model = mx.symbol.dot(lhs=x, rhs=v)
         y = mx.symbol.Variable("label")
         model = mx.symbol.LinearRegressionOutput(data=model, label=y, name="out")
         return model

    n, m = 128, 100
    model = regression_model(m)

    data = mx.nd.zeros(shape=(n, m), stype='csr')
    label = mx.nd.zeros((n, 1))
    iterator = mx.io.NDArrayIter(data=data, label={'label':label},
                                 batch_size=n, last_batch_handle='discard')

    # create module
    mod = mx.mod.Module(symbol=model, data_names=['data'], label_names=['label'])
    mod.bind(data_shapes=iterator.provide_data, label_shapes=iterator.provide_label)
    mod.init_params()
    v = mod._arg_params['v']
    assert(v.stype == 'row_sparse')
    assert(np.sum(v.asnumpy()) != 0)

@with_seed()
def test_forward_reshape():
    num_class=10
    data1 = mx.sym.Variable('data1')
    data2 = mx.sym.Variable('data2')
    conv1 = mx.sym.Convolution(data=data1, kernel=(2, 2), num_filter=2, stride=(2, 2))
    conv2 = mx.sym.Convolution(data=data2, kernel=(3, 3), num_filter=3, stride=(1, 1))
    pooling1 = mx.sym.Pooling(data=conv1, kernel=(2, 2), stride=(1, 1), pool_type="avg")
    pooling2 = mx.sym.Pooling(data=conv2, kernel=(2, 2), stride=(1, 1), pool_type="max")
    flatten1 = mx.sym.flatten(data=pooling1)
    flatten2 = mx.sym.flatten(data=pooling2)
    sum = mx.sym.sum(data=flatten1, axis=1) + mx.sym.sum(data=flatten2, axis=1)
    fc = mx.sym.FullyConnected(data=sum, num_hidden=num_class)
    sym = mx.sym.SoftmaxOutput(data=fc, name='softmax')

    dshape1 = (10, 3, 64, 64)
    dshape2 = (10, 3, 32, 32)
    lshape = (10,)

    mod = mx.mod.Module(symbol=sym, data_names=['data1', 'data2'],
                        label_names=['softmax_label'])
    mod.bind(data_shapes=[('data1', dshape1), ('data2', dshape2)],
             label_shapes=[('softmax_label', lshape)])
    mod.init_params()
    mod.init_optimizer(optimizer_params={'learning_rate': 0.01})

    # Train with original data shapes
    data_batch = mx.io.DataBatch(data=[mx.nd.random.uniform(0, 9, dshape1),
                                       mx.nd.random.uniform(5, 15, dshape2)],
                                 label=[mx.nd.ones(lshape)])
    mod.forward(data_batch)
    assert mod.get_outputs()[0].shape == tuple([lshape[0], num_class])
    mod.backward()
    mod.update()

    # Train with different batch size
    dshape1 = (3, 3, 64, 64)
    dshape2 = (3, 3, 32, 32)
    lshape = (3,)
    data_batch = mx.io.DataBatch(data=[mx.nd.random.uniform(0, 9, dshape1),
                                       mx.nd.random.uniform(5, 15, dshape2)],
                                 label=[mx.nd.ones(lshape)])
    mod.forward(data_batch)
    assert mod.get_outputs()[0].shape == tuple([lshape[0], num_class])
    mod.backward()
    mod.update()

    dshape1 = (20, 3, 64, 64)
    dshape2 = (20, 3, 32, 32)
    lshape = (20,)
    data_batch = mx.io.DataBatch(data=[mx.nd.random.uniform(3, 5, dshape1),
                                       mx.nd.random.uniform(10, 25, dshape2)],
                                 label=[mx.nd.ones(lshape)])
    mod.forward(data_batch)
    assert mod.get_outputs()[0].shape == tuple([lshape[0], num_class])
    mod.backward()
    mod.update()

    #Train with both different batch size and data shapes
    dshape1 = (20, 3, 120, 120)
    dshape2 = (20, 3, 32, 64)
    lshape = (20,)
    data_batch = mx.io.DataBatch(data=[mx.nd.random.uniform(0, 9, dshape1),
                                       mx.nd.random.uniform(5, 15, dshape2)],
                                 label=[mx.nd.ones(lshape)])
    mod.forward(data_batch)
    assert mod.get_outputs()[0].shape == tuple([lshape[0], num_class])
    mod.backward()
    mod.update()

    dshape1 = (5, 3, 28, 40)
    dshape2 = (5, 3, 24, 16)
    lshape = (5,)
    data_batch = mx.io.DataBatch(data=[mx.nd.random.uniform(0, 9, dshape1),
                                       mx.nd.random.uniform(15, 25, dshape2)],
                                 label=[mx.nd.ones(lshape)])
    mod.forward(data_batch)
    assert mod.get_outputs()[0].shape == tuple([lshape[0], num_class])
    mod.backward()
    mod.update()

    #Test score
    dataset_shape1 = (30, 3, 30, 30)
    dataset_shape2 = (30, 3, 20, 40)
    labelset_shape = (30,)

    eval_dataiter = mx.io.NDArrayIter(data=[mx.nd.random.uniform(0, 9, dataset_shape1),
                                            mx.nd.random.uniform(15, 25, dataset_shape2)],
                                      label=[mx.nd.ones(labelset_shape)],
                                      batch_size=5)
    assert len(mod.score(eval_data=eval_dataiter, eval_metric='acc')) == 1

    #Test prediction
    dshape1 = (1, 3, 30, 30)
    dshape2 = (1, 3, 20, 40)
    dataset_shape1 = (10, 3, 30, 30)
    dataset_shape2 = (10, 3, 20, 40)

    pred_dataiter = mx.io.NDArrayIter(data=[mx.nd.random.uniform(0, 9, dataset_shape1),
                                            mx.nd.random.uniform(15, 25, dataset_shape2)])
    mod.bind(data_shapes=[('data1', dshape1), ('data2', dshape2)],
             for_training=False, force_rebind=True)
    assert mod.predict(pred_dataiter).shape == tuple([10, num_class])

@with_seed()
def test_forward_types():
    #Test forward with other data batch API
    Batch = namedtuple('Batch', ['data'])
    data = mx.sym.Variable('data')
    out = data * 2
    mod = mx.mod.Module(symbol=out, label_names=None)
    mod.bind(data_shapes=[('data', (1, 10))])
    mod.init_params()
    data1 = [mx.nd.ones((1, 10))]
    mod.forward(Batch(data1))
    assert mod.get_outputs()[0].shape == (1, 10)
    data2 = [mx.nd.ones((3, 5))]
    mod.forward(Batch(data2))
    assert mod.get_outputs()[0].shape == (3, 5)

    #Test forward with other NDArray and np.ndarray inputs
    data = mx.sym.Variable('data')
    out = data * 2
    mod = mx.mod.Module(symbol=out, label_names=None)
    mod.bind(data_shapes=[('data', (1, 10))])
    mod.init_params()
    data1 = mx.nd.ones((1, 10))
    assert mod.predict(data1).shape == (1, 10)
    data2 = np.ones((1, 10))
    assert mod.predict(data1).shape == (1, 10)


def test_reference_single_batch_during_fit():
    """
    When using C++-based iterators, it's important that only a single batch is referenced at a time. Because C++
    iterators are exposed to the Python code through a C API, there is no concept of reference counting. Hence,
    typically C++ iterators will deallocate a batch when next() is called on them. So, we need to make sure the Python
    code only references a single batch at a time, otherwise the Python code will attempt to access freed memory,
    resulting in either (a) garbage accuracy or (b) a segmentation fault.
    """
    current_batch_i = None

    class MockBatch(object):
        def __init__(self, i):
            self.i = i

        @property
        def label(self):
            global current_batch_i
            assert self.i == current_batch_i

    class MockTrainData(object):
        def __init__(self, batches):
            self._i = 0
            self._batches = batches
            self.provide_data = None
            self.provide_label = None
            self.reset = lambda: None

        def __iter__(self):
            self._i = 0
            return self

        def __next__(self):
            global current_batch_i

            if self._i < self._batches:
                current_batch_i = self._i
                self._i += 1
                return MockBatch(current_batch_i)
            raise StopIteration

        def next(self):
            return self.__next__()

    mod = mx.mod.BaseModule()

    def empty_fn(*args, **kwargs):
        pass
    mod.bind = empty_fn
    mod.init_params = empty_fn
    mod.init_optimizer = empty_fn
    mod.forward = empty_fn
    mod.backward = empty_fn
    mod.update = empty_fn
    mod.update_metric = empty_fn
    mod.get_params = lambda: (None, None)

    train_data = MockTrainData(batches=2)
    mod.fit(train_data, num_epoch=1)

@with_seed()
def test_bucket_module_grad_req():
    batch_size = 2
    def sym_gen(_):
        data = mx.symbol.Variable('data')
        weight = mx.symbol.Variable('a', shape=(1,), init=mx.init.One())
        sym = mx.sym.make_loss(mx.sym.broadcast_mul(data, weight))
        return sym, ('data',), None

    mod = mx.mod.BucketingModule(sym_gen=sym_gen, default_bucket_key=10)
    mod.bind(data_shapes=[['data', (batch_size, )]], for_training=True, grad_req='write')
    mod.init_params()

    mod.forward_backward(mx.io.DataBatch(data=[mx.nd.ones((batch_size,))],
                                         label=None,
                                         provide_data=[mx.io.DataDesc(name='data', shape=(batch_size, ), layout='N')],
                                         bucket_key=10))
    assert(mod._curr_module._exec_group.execs[0].grad_dict['a'].asscalar() == batch_size)

    mod.forward_backward(mx.io.DataBatch(data=[mx.nd.ones((batch_size,))],
                                         label=None,
                                         provide_data=[mx.io.DataDesc(name='data', shape=(batch_size, ), layout='N')],
                                         bucket_key=5))
    assert(mod._curr_module._exec_group.execs[0].grad_dict['a'].asscalar() == batch_size)

    mod = mx.mod.BucketingModule(sym_gen=sym_gen, default_bucket_key=10)
    mod.bind(data_shapes=[['data', (batch_size, )]], for_training=True, grad_req='add')
    mod.init_params()

    mod.forward_backward(mx.io.DataBatch(data=[mx.nd.ones((batch_size,))],
                                         label=None,
                                         provide_data=[mx.io.DataDesc(name='data', shape=(batch_size,), layout='N')],
                                         bucket_key=10))
    assert(mod._curr_module._exec_group.execs[0].grad_dict['a'].asscalar() == batch_size)

    mod.forward_backward(mx.io.DataBatch(data=[mx.nd.ones((batch_size,))],
                                         label=None,
                                         provide_data=[mx.io.DataDesc(name='data', shape=(batch_size,), layout='N')],
                                         bucket_key=5))
    assert mod._curr_module._grad_req == 'add'
    assert(mod._curr_module._exec_group.execs[0].grad_dict['a'].asscalar() == 2 * batch_size)


def test_module_update_no_pragram():
    # test module to do update on layers without params
    data_shape = (10, 10)
    data = mx.sym.Variable('data')
    out = mx.sym.Dropout(data, 0.5)
    mod = mx.mod.Module(out)
    mod.bind(data_shapes=[('data', data_shape)])
    mod.init_params()
    mod.init_optimizer()
    data_batch = mx.io.DataBatch([nd.ones(data_shape)])
    mod.forward_backward(data_batch)
    mod.update()
    assert(mod.get_outputs()[0].shape == data_shape)


def test_module_init_optimizer():
    def get_module_idx2name(mod):
        idx2name = {}
        idx2name.update(enumerate(mod._exec_group.param_names))
        return idx2name

    data = mx.sym.Variable('data')
    sym = mx.sym.FullyConnected(data, num_hidden=20, name='fc')
    batch_size = 8
    opt_params = {'learning_rate': 1, 'rescale_grad': 1.0 / batch_size}

    # Pass an optimizer str
    mod1 = mx.mod.Module(sym, ('data',), None, context=mx.cpu(0))
    mod1.bind(data_shapes=[('data', (batch_size, 20))])
    mod1.init_params()
    mod1.init_optimizer(optimizer='sgd', optimizer_params=opt_params)
    assert mod1._optimizer.idx2name == get_module_idx2name(mod1)

    # Pass an Optimizer object
    mod2 = mx.mod.Module(sym, ('data',), None, context=mx.cpu(0))
    mod2.bind(data_shapes=[('data', (batch_size, 20))])
    mod2.init_params()
    opt = mx.optimizer.SGD(**opt_params)
    mod2.init_optimizer(optimizer=opt)
    assert mod2._optimizer.idx2name == get_module_idx2name(mod2)

