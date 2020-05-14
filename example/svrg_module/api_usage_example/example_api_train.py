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
from mxnet.contrib.svrg_optimization.svrg_module import SVRGModule


def test_svrg_intermediate_level_api(args):
    """Demonstrates intermediate level SVRGModule API where the training process
    need to be explicitly defined. KVstore is not explicitly created.

    Parameters
    ----------
    args: args
        Command line arguments
    """
    num_epoch = args.epochs
    batch_size = args.batch_size
    update_freq = args.update_freq

    di, mod = create_network(batch_size, update_freq)

    mod.bind(data_shapes=di.provide_data, label_shapes=di.provide_label)
    mod.init_params(initializer=mx.init.Uniform(0.01), allow_missing=False, force_init=False, allow_extra=False)
    kv = mx.kv.create("local")
    mod.init_optimizer(kvstore=kv, optimizer='sgd', optimizer_params=(('learning_rate', 0.025),))
    metrics = mx.metric.create("mse")
    for e in range(num_epoch):
        metrics.reset()
        if e % mod.update_freq == 0:
            mod.update_full_grads(di)
        di.reset()
        for batch in di:
            mod.forward_backward(data_batch=batch)
            mod.update()
            mod.update_metric(metrics, batch.label)
        mod.logger.info('Epoch[%d] Train cost=%f', e, metrics.get()[1])


def test_svrg_high_level_api(args):
    """Demonstrates suggested usage of  high level SVRGModule API. KVStore is explicitly created.

    Parameters
    ----------
    args: args
        Command line arguments
    """
    num_epoch = args.epochs
    batch_size = args.batch_size
    update_freq = args.update_freq

    di, mod = create_network(batch_size, update_freq)
    mod.fit(di, eval_metric='mse', optimizer='sgd', optimizer_params=(('learning_rate', 0.025),), num_epoch=num_epoch,
            kvstore='local')


def create_network(batch_size, update_freq):
    """Create a linear regression network for performing SVRG optimization.
    Parameters
    ----------
    batch_size: int
        Size of data split
    update_freq: int
        Update Frequency for calculating full gradients

    Returns
    ----------
    di: mx.io.NDArrayIter
        Data iterator
    update_freq: SVRGModule
        An instance of SVRGModule for performing SVRG optimization
    """
    import logging
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(level=logging.INFO, format=head)

    train_data = np.random.randint(1, 5, [1000, 2])
    weights = np.array([1.0, 2.0])
    train_label = train_data.dot(weights)

    di = mx.io.NDArrayIter(train_data, train_label, batch_size=batch_size, shuffle=True, label_name='lin_reg_label')
    X = mx.sym.Variable('data')
    Y = mx.symbol.Variable('lin_reg_label')
    fully_connected_layer = mx.sym.FullyConnected(data=X, name='fc1', num_hidden=1)
    lro = mx.sym.LinearRegressionOutput(data=fully_connected_layer, label=Y, name="lro")

    mod = SVRGModule(
        symbol=lro,
        data_names=['data'],
        label_names=['lin_reg_label'], update_freq=update_freq, logger=logging
    )

    return di, mod

# run as a script
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-e', dest='epochs', default=100, type=int)
    parser.add_argument('-bs', dest='batch_size', default=32, type=int)
    parser.add_argument('-f', dest="update_freq", default=2, type=int)
    args = parser.parse_args()

    print("========================== Intermediate Level API ==========================")
    test_svrg_intermediate_level_api(args)
    print("========================== High Level API ==========================")
    test_svrg_high_level_api(args)
