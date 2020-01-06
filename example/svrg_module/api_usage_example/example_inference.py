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
import logging
from mxnet.contrib.svrg_optimization.svrg_module import SVRGModule


def test_svrg_inference(args):
    epoch = args.epochs
    batch_size = args.batch_size
    update_freq = args.update_freq

    train_iter, val_iter, mod = create_network(batch_size, update_freq)
    mod.fit(train_iter, eval_data=val_iter, eval_metric='mse', optimizer='sgd',
            optimizer_params=(('learning_rate', 0.025),),
            num_epoch=epoch)


def get_validation_score(args):
    epoch = args.epochs
    batch_size = args.batch_size
    update_freq = args.update_freq

    train_iter, val_iter,  mod = create_network(batch_size, update_freq)
    mod.bind(data_shapes=train_iter.provide_data, label_shapes=train_iter.provide_label)
    mod.init_params(initializer=mx.init.Uniform(0.01), allow_missing=False, force_init=False, allow_extra=False)
    mod.init_optimizer(kvstore='local', optimizer='sgd', optimizer_params=(('learning_rate', 0.025),))
    metrics = mx.metric.create("mse")
    for e in range(epoch):
        metrics.reset()
        if e % mod.update_freq == 0:
            mod.update_full_grads(train_iter)
        train_iter.reset()
        for batch in train_iter:
            mod.forward_backward(data_batch=batch)
            mod.update()
            mod.update_metric(metrics, batch.label)

    y = mod.predict(val_iter)

    # test-train data split, 20% test data out of 1000 data samples
    assert y.shape == (200, 1)
    score = mod.score(val_iter, ['mse'])
    print("Training Loss on Validation Set is {}".format(score[0][1]))


def create_network(batch_size, update_freq):
    """Create a linear regression network for performing SVRG optimization.
    :return: an instance of mx.io.NDArrayIter
    :return: an instance of mx.mod.svrgmodule for performing SVRG optimization
    """
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(level=logging.INFO, format=head)
    data = np.random.randint(1, 5, [1000, 2])

    #Test_Train data split
    n_train = int(data.shape[0] * 0.8)
    weights = np.array([1.0, 2.0])
    label = data.dot(weights)

    di = mx.io.NDArrayIter(data[:n_train, :], label[:n_train], batch_size=batch_size, shuffle=True, label_name='lin_reg_label')
    val_iter = mx.io.NDArrayIter(data[n_train:, :], label[n_train:], batch_size=batch_size)

    X = mx.sym.Variable('data')
    Y = mx.symbol.Variable('lin_reg_label')
    fully_connected_layer = mx.sym.FullyConnected(data=X, name='fc1', num_hidden=1)
    lro = mx.sym.LinearRegressionOutput(data=fully_connected_layer, label=Y, name="lro")

    mod = SVRGModule(
        symbol=lro,
        data_names=['data'],
        label_names=['lin_reg_label'], update_freq=update_freq, logger=logging)

    return di, val_iter, mod


# run as a script
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', dest='epochs', default=100, type=int)
    parser.add_argument('-bs', dest='batch_size', default=32, type=int)
    parser.add_argument('-f', dest="update_freq", default=2, type=int)
    args = parser.parse_args()

    print("========================== SVRG Module Inference ==========================")
    test_svrg_inference(args)
    print("========================SVRG Module Score ============================")
    get_validation_score(args)
