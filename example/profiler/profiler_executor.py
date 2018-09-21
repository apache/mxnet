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
import argparse
import os
import time
import numpy as np
from mxnet import profiler
import memonger


def parse_args():
    parser = argparse.ArgumentParser(description='Set network parameters for benchmark test.')
    parser.add_argument('--profile_filename', type=str, default='profile_executor_5iter.json')
    parser.add_argument('--iter_num', type=int, default=5)
    parser.add_argument('--fc1', type=int, default=128)
    parser.add_argument('--fc2', type=int, default=128)
    parser.add_argument('--fc3', type=int, default=128)
    parser.add_argument('--fc4', type=int, default=128)
    return parser.parse_args()


def _download(data_dir):
    if not os.path.isdir(data_dir):
        os.system("mkdir " + data_dir)
    os.chdir(data_dir)
    if (not os.path.exists('train-images-idx3-ubyte')) or \
       (not os.path.exists('train-labels-idx1-ubyte')) or \
       (not os.path.exists('t10k-images-idx3-ubyte')) or \
       (not os.path.exists('t10k-labels-idx1-ubyte')):
        os.system("wget http://webdocs.cs.ualberta.ca/~bx3/data/mnist.zip")
        os.system("unzip -u mnist.zip; rm mnist.zip")
    os.chdir("..")


def get_data(data_shape):
    data_dir = "mnist/"
    batch_size = 128
    if '://' not in data_dir:
        _download(data_dir)

    train           = mx.io.MNISTIter(
        image       = data_dir + "train-images-idx3-ubyte",
        label       = data_dir + "train-labels-idx1-ubyte",
        input_shape = data_shape,
        batch_size  = batch_size,
        shuffle     = True,
        )

    val = mx.io.MNISTIter(
        image       = data_dir + "t10k-images-idx3-ubyte",
        label       = data_dir + "t10k-labels-idx1-ubyte",
        input_shape = data_shape,
        batch_size  = batch_size,
        )

    return (train, val)

def get_symbol():
    data = mx.symbol.Variable('data')
    fc1  = mx.symbol.FullyConnected(data=data, name='fc1', num_hidden=args.fc1)
    act1 = mx.symbol.Activation(data=fc1, name='relu1', act_type='relu')
    fc2  = mx.symbol.FullyConnected(data=act1 , name='fc2', num_hidden=args.fc2)
    act2 = mx.symbol.Activation(data=fc2, name='relu2', act_type='relu')
    fc3  = mx.symbol.FullyConnected(data=act2 , name='fc3', num_hidden=args.fc3)
    act3 = mx.symbol.Activation(data=fc3, name='relu3', act_type='relu')
    fc4  = mx.symbol.FullyConnected(data=act3 , name='fc4', num_hidden=args.fc4)
    act4 = mx.symbol.Activation(data=fc4, name='relu4', act_type='relu')
    fc5  = mx.symbol.FullyConnected(data=act4 , name='fc5', num_hidden=10)
    net  = mx.symbol.SoftmaxOutput(data=fc5 , name='softmax')
    return net, [('data', (128, 1, 28, 28))], [('softmax_label', (128, ))]

def get_module(ctx, sym, provide_data, provide_label, batch_size=None, is_train=True, use_memonger=False):
    if use_memonger:
        name, data_shapes = provide_data[0]
        sym = memonger.search_plan(sym, data=data_shapes)
    mod = mx.mod.Module(symbol=sym,
                        data_names=[name for name, _ in provide_data],
                        label_names=[name for name, _ in provide_label],
                        context=ctx)
    if batch_size is not None:
        provide_data = [(name, (batch_size,) + shape[1:]) for name, shape in provide_data]
        provide_label = [(name, (batch_size,) + shape[1:]) for name, shape in provide_label]
    if is_train:
        mod.bind(data_shapes=provide_data, label_shapes=provide_label, for_training=True, inputs_need_grad=False)
    else:
        mod.bind(data_shapes=provide_data, label_shapes=provide_label, for_training=False, inputs_need_grad=False)

    mod.init_params(initializer=mx.init.Xavier(magnitude=2.))
    mod.init_optimizer(optimizer='ccsgd',
                       optimizer_params={
                            'learning_rate': 0.0001,
                            'momentum': 0.0,
                            'wd': 0.0
                        })
    return mod


def benchmark(mod, dry_run=10, iterations=10):
    if len(mod._context) == 1:
        ctx = mod._context[0]
    else:
        ctx = mx.cpu()
    data = [mx.random.uniform(-1.0, 1.0, shape=shape, ctx=ctx) for _, shape in mod.data_shapes]
    label = [mx.nd.array(np.random.randint(1, 100, size=shape), ctx=ctx) for _, shape in mod.label_shapes]
    batch = mx.io.DataBatch(data, label)

    # dry run
    for i in range(dry_run):
        mod.forward(batch, is_train=True)
        mod.backward()
        for output in mod.get_outputs(merge_multi_context=False)[0]:
            output.wait_to_read()
        mod.update()

    t0 = time.clock()

    profiler.set_state('run')
    # real run
    for i in range(iterations):
        mod.forward(batch, is_train=True)
        mod.backward()
        mod.update()
        for output in mod.get_outputs(merge_multi_context=False)[0]:
            output.wait_to_read()
    profiler.set_state('stop')

    t1 = time.clock()
    return (t1 - t0)*1000.0 / iterations


def executor(num_iteration):
    sym, provide_data, provide_label = get_symbol()
    ctx = [mx.cpu(0)]
    mod = get_module(ctx, sym, provide_data, provide_label, batch_size=128)
    return benchmark(mod, iterations=args.iter_num)


args = parse_args()

if __name__ == '__main__':
    mx.profiler.set_config(profile_symbolic=True, filename=args.profile_filename)
    print('profile file save to {0}'.format(args.profile_filename))
    print('executor num_iteration: {0}'.format(args.iter_num))
    executor_time = executor(args.iter_num)
    print("executor {0} ms / iteration".format(executor_time))
