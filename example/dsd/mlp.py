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
import os
import logging
import argparse
from math import ceil
import sparse_sgd

# symbol net
def get_symbol():
    data = mx.symbol.Variable('data')
    fc1 = mx.symbol.FullyConnected(data, name='fc1', num_hidden=128)
    act1 = mx.symbol.Activation(fc1, name='relu1', act_type="relu")
    fc2 = mx.symbol.FullyConnected(act1, name='fc2', num_hidden=64)
    act2 = mx.symbol.Activation(fc2, name='relu2', act_type="relu")
    fc3 = mx.symbol.FullyConnected(act2, name='fc3', num_hidden=10)
    softmax = mx.symbol.SoftmaxOutput(fc3, name='sm')

    return softmax

# download ubyte version of mnist and untar
def download_data():
    if not os.path.isdir("data/"):
        os.system("mkdir data/")
    if (not os.path.exists('data/train-images-idx3-ubyte')) or \
       (not os.path.exists('data/train-labels-idx1-ubyte')) or \
       (not os.path.exists('data/t10k-images-idx3-ubyte')) or \
       (not os.path.exists('data/t10k-labels-idx1-ubyte')):
        os.system("wget -q http://data.mxnet.io/mxnet/data/mnist.zip -P data/")
        os.chdir("./data")
        os.system("unzip -u mnist.zip")
        os.chdir("..")

# get data iterators
def get_iters(batch_size):
    train = mx.io.MNISTIter(
        image="data/train-images-idx3-ubyte",
        label="data/train-labels-idx1-ubyte",
        data_shape=(784,),
        label_name='sm_label',
        batch_size=batch_size,
        shuffle=True,
        flat=True,
        silent=False,
        seed=10)
    val = mx.io.MNISTIter(
        image="data/t10k-images-idx3-ubyte",
        label="data/t10k-labels-idx1-ubyte",
        data_shape=(784,),
        label_name='sm_label',
        batch_size=batch_size,
        shuffle=True,
        flat=True,
        silent=False)

    return (train, val)

def test_mlp(args):
    # get parameters
    prefix = './mlp'
    batch_size = 100
    pruning_switch_epoch = [int(i) for i in args.pruning_switch_epoch.split(',')]
    num_epoch = pruning_switch_epoch[-1]
    batches_per_epoch = ceil(60000.0/batch_size)
    weight_sparsity = args.weight_sparsity
    bias_sparsity = args.bias_sparsity
    weight_threshold = args.weight_threshold
    bias_threshold = args.bias_threshold
    if args.weight_sparsity:
        weight_sparsity = [float(i) for i in args.weight_sparsity.split(',')]
        bias_sparsity = [float(i) for i in args.bias_sparsity.split(',')]
    else:
        weight_threshold = [float(i) for i in args.weight_threshold.split(',')]
        bias_threshold = [float(i) for i in args.bias_threshold.split(',')]

    # get symbols and iterators
    sym = get_symbol()
    download_data()
    (train, val) = get_iters(batch_size)

    # fit model
    model = mx.mod.Module(
        sym,
        context=[mx.cpu(i) for i in range(2)],
        data_names=['data'],
        label_names=['sm_label'])
    optimizer_params = {
        'learning_rate'             : 0.1,
        'wd'                        : 0.004,
        'momentum'                  : 0.9,
        'pruning_switch_epoch'      : pruning_switch_epoch,
        'batches_per_epoch'         : batches_per_epoch,
        'weight_sparsity'           : weight_sparsity,
        'bias_sparsity'             : bias_sparsity,
        'weight_threshold'          : weight_threshold,
        'bias_threshold'            : bias_threshold}
    logging.info('Start training...')
    model.fit(train,
        eval_data=val,
        eval_metric='acc',
        epoch_end_callback=mx.callback.do_checkpoint(prefix),
        num_epoch=num_epoch,
        optimizer='sparsesgd',
        optimizer_params=optimizer_params)
    logging.info('Finish traning...')

    # remove files
    for i in range(num_epoch):
        os.remove('%s-%04d.params' % (prefix, i + 1))
    os.remove('%s-symbol.json' % prefix)


if __name__ == "__main__":

    # print logging by default
    logging.basicConfig(level=logging.DEBUG)

    parser = argparse.ArgumentParser(description="sparse training")
    parser.add_argument('--pruning_switch_epoch', type=str)
    parser.add_argument('--weight_sparsity', type=str, default=None)
    parser.add_argument('--bias_sparsity', type=str, default=None)
    parser.add_argument('--weight_threshold', type=str, default=None)
    parser.add_argument('--bias_threshold', type=str, default=None)
    args = parser.parse_args()

    test_mlp(args)
