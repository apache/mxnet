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
import os, sys
from utils import get_data
import mxnet as mx
import numpy as np
import logging

# whether to demo model-parallelism + data parallelism
demo_data_model_parallelism = True

if demo_data_model_parallelism:
    contexts = [[mx.context.gpu(0), mx.context.gpu(1)], [mx.context.gpu(2), mx.context.gpu(3)]]
else:
    contexts = [mx.context.cpu(), mx.context.cpu()]

#--------------------------------------------------------------------------------
# module 1
#--------------------------------------------------------------------------------
data = mx.symbol.Variable('data')
fc1 = mx.symbol.FullyConnected(data, name='fc1', num_hidden=128)
act1 = mx.symbol.Activation(fc1, name='relu1', act_type="relu")

mod1 = mx.mod.Module(act1, label_names=[], context=contexts[0])

#--------------------------------------------------------------------------------
# module 2
#--------------------------------------------------------------------------------
data = mx.symbol.Variable('data')
fc2 = mx.symbol.FullyConnected(data, name = 'fc2', num_hidden = 64)
act2 = mx.symbol.Activation(fc2, name='relu2', act_type="relu")
fc3 = mx.symbol.FullyConnected(act2, name='fc3', num_hidden=10)
softmax = mx.symbol.SoftmaxOutput(fc3, name = 'softmax')

mod2 = mx.mod.Module(softmax, context=contexts[1])

#--------------------------------------------------------------------------------
# Container module
#--------------------------------------------------------------------------------
mod_seq = mx.mod.SequentialModule()
mod_seq.add(mod1).add(mod2, take_labels=True, auto_wiring=True)


#--------------------------------------------------------------------------------
# Training
#--------------------------------------------------------------------------------
n_epoch = 2
batch_size = 100
basedir = os.path.dirname(__file__)
get_data.get_mnist(os.path.join(basedir, "data"))

train_dataiter = mx.io.MNISTIter(
        image=os.path.join(basedir, "data", "train-images-idx3-ubyte"),
        label=os.path.join(basedir, "data", "train-labels-idx1-ubyte"),
        data_shape=(784,),
        batch_size=batch_size, shuffle=True, flat=True, silent=False, seed=10)
val_dataiter = mx.io.MNISTIter(
        image=os.path.join(basedir, "data", "t10k-images-idx3-ubyte"),
        label=os.path.join(basedir, "data", "t10k-labels-idx1-ubyte"),
        data_shape=(784,),
        batch_size=batch_size, shuffle=True, flat=True, silent=False)

logging.basicConfig(level=logging.DEBUG)
mod_seq.fit(train_dataiter, eval_data=val_dataiter,
            optimizer_params={'learning_rate':0.01, 'momentum': 0.9}, num_epoch=n_epoch)

