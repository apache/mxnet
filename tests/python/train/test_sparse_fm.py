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
import mxnet.ndarray as nd
from mxnet.test_utils import *
import numpy as np

def test_factorization_machine_module(verbose=False):
    """ Test factorization machine model with sparse operators """
    def check_factorization_machine_module(optimizer=None, num_epochs=None):
        print("check_factorization_machine_module( {} )".format(optimizer))

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
        if optimizer == 'sgd':
            # use Sparse SGD with learning rate 0.1 to train
            sgd = mx.optimizer.SGD(momentum=0.1, clip_gradient=5.0, learning_rate=0.01,
                                   rescale_grad=1.0/batch_size)
            mod.init_optimizer(optimizer=sgd)
            if num_epochs is None:
                num_epochs = 10
            expected_accuracy = 0.02
        elif optimizer == 'adam':
            # use Sparse Adam to train
            adam = mx.optimizer.Adam(clip_gradient=5.0, learning_rate=0.0005,
                                     rescale_grad=1.0/batch_size)
            mod.init_optimizer(optimizer=adam)
            if num_epochs is None:
                num_epochs = 10
            expected_accuracy = 0.05
        elif optimizer == 'adagrad':
            # use Sparse AdaGrad with learning rate 0.1 to train
            adagrad = mx.optimizer.AdaGrad(clip_gradient=5.0, learning_rate=0.01,
                                           rescale_grad=1.0/batch_size)
            mod.init_optimizer(optimizer=adagrad)
            if num_epochs is None:
                num_epochs = 20
            expected_accuracy = 0.09
        else:
            raise AssertionError("Unsupported optimizer type '" + optimizer + "' specified")
        # use accuracy as the metric
        metric = mx.metric.create('MSE')
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

    if verbose is True:
        print("============ SGD ==========================")
        start = time.clock()
    check_factorization_machine_module('sgd')
    if verbose is True:
        print("Duration: {}".format(time.clock() - start))
        print("============ ADAM ==========================")
        start = time.clock()
    check_factorization_machine_module('adam')
    if verbose is True:
        print("Duration: {}".format(time.clock() - start))
        print("============ ADAGRAD ==========================")
        start = time.clock()
    check_factorization_machine_module('adagrad')
    if verbose is True:
        print("Duration: {}".format(time.clock() - start))

# run as a script
if __name__ == "__main__":
    test_factorization_machine_module()	
