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

# pylint: skip-file
import numpy as np
import mxnet as mx
import numba
import logging

# We use numba.jit to implement the loss gradient.
@numba.jit
def mc_hinge_grad(scores, labels):
    scores = scores.asnumpy()
    labels = labels.asnumpy().astype(int)

    n, _ = scores.shape
    grad = np.zeros_like(scores)

    for i in range(n):
        score = 1 + scores[i] - scores[i, labels[i]]
        score[labels[i]] = 0
        ind_pred = score.argmax()
        grad[i, labels[i]] -= 1
        grad[i, ind_pred] += 1

    return grad

if __name__ == '__main__':
    n_epoch = 10
    batch_size = 100
    num_gpu = 2
    contexts = mx.context.cpu() if num_gpu < 1 else [mx.context.gpu(i) for i in range(num_gpu)]

    # build a MLP module
    data = mx.symbol.Variable('data')
    fc1 = mx.symbol.FullyConnected(data, name='fc1', num_hidden=128)
    act1 = mx.symbol.Activation(fc1, name='relu1', act_type="relu")
    fc2 = mx.symbol.FullyConnected(act1, name = 'fc2', num_hidden = 64)
    act2 = mx.symbol.Activation(fc2, name='relu2', act_type="relu")
    fc3 = mx.symbol.FullyConnected(act2, name='fc3', num_hidden=10)

    mlp = mx.mod.Module(fc3, context=contexts)
    loss = mx.mod.PythonLossModule(grad_func=mc_hinge_grad)

    mod = mx.mod.SequentialModule() \
            .add(mlp) \
            .add(loss, take_labels=True, auto_wiring=True)

    train_dataiter = mx.io.MNISTIter(
            image="data/train-images-idx3-ubyte",
            label="data/train-labels-idx1-ubyte",
            data_shape=(784,),
            batch_size=batch_size, shuffle=True, flat=True, silent=False, seed=10)
    val_dataiter = mx.io.MNISTIter(
            image="data/t10k-images-idx3-ubyte",
            label="data/t10k-labels-idx1-ubyte",
            data_shape=(784,),
            batch_size=batch_size, shuffle=True, flat=True, silent=False)

    logging.basicConfig(level=logging.DEBUG)
    mod.fit(train_dataiter, eval_data=val_dataiter,
            optimizer_params={'learning_rate':0.01, 'momentum': 0.9},
            num_epoch=n_epoch)
