#!/usr/bin/env python3

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

# coding: utf-8
# pylint: disable=arguments-differ

# This test checks if dynamic loading of library into MXNet is successful
# and checks the end of end computation of custom operator

import os, ctypes
import mxnet as mx
import time
from mxnet.gluon import nn
from mxnet import nd
import numpy as np
from mxnet.lr_scheduler import PolyScheduler
from mxnet import np, npx
from pos_trainer import POS_Trainer
try:
    import horovod.mxnet as hvd
except ImportError:
    pass
from mxnet.base import _LIB, check_call, mx_uint, c_str, c_str_array, SymbolHandle

# load library
if (os.name=='posix'):
    path = os.path.abspath('add_reduce_op_lib.so')
    mx.library.load(path)
elif (os.name=='nt'):
    path = os.path.abspath('add_reduce_op_lib.dll')
    mx.library.load(path)


class Easynet(nn.HybridBlock):
    def __init__(self, n):
        super().__init__()
        self.ls = nn.HybridSequential()
        for i in range(n):
            self.ls.add(nn.Dense(in_units=2, units=2, flatten=False))



    def forward(self, input):
        input = self.ls(input)
        return input


def test_model():
    from mxnet import gluon
    from mxnet.gluon import Block, nn, HybridBlock
    from mxnet import init



    hvd.init()
    rank = hvd.rank()
    size = hvd.size()
    ctx = mx.gpu(rank)

    np.random.seed(1234 + 10 * rank)
    mx.random.seed(1234 + 10 * rank)


    number = 2
    model = Easynet(number)

    if rank == 0:
        for i in range(number):
            model.ls[i].weight.initialize(init=init.One(), ctx=ctx)
            model.ls[i].bias.initialize(init=init.One(), ctx=ctx)
    else:
        for i in range(number):
            model.ls[i].weight.initialize(init=init.Zero(), ctx=ctx)
            model.ls[i].bias.initialize(init=init.Zero(), ctx=ctx)

    model.hybridize()

    params = model.collect_params()
    lr_scheduler = PolyScheduler(max_update=1,
                                 base_lr=1e-3,
                                 warmup_begin_lr=0.0,
                                 pwr=1,
                                 final_lr=0.0,
                                 warmup_steps=0,
                                 warmup_mode='linear')
    optimizer_params = {'learning_rate': 1e-3,
                        'wd': 1e-2,
                        'lr_scheduler': lr_scheduler}
    trainer = POS_Trainer(params, "adam", optimizer_params)

    options = trainer.generate_graph_pass_options()
    backward_options = trainer.generate_backward_options()
    x = np.ones((1,2), ctx = ctx)
    label = np.ones((2, ), ctx = ctx) * rank
    #print(options)

    loss_function = gluon.loss.L2Loss()

    model.optimize_for(x, backend = "add_reduce_op", **options)
    for i in range(1):
        with mx.autograd.record():
            out = model(x)
            loss = loss_function(out, label).mean() / size
            loss.backward(backward_option = backward_options)
            mx.npx.waitall()
            mx.nd.waitall()

    for name in params:
        print(name, params[name].list_grad()[0])
    print('finish')


test_model()
