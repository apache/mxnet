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
#from gluonnlp.layers import get_activation
from mxnet.base import _LIB, check_call, mx_uint, c_str, c_str_array, SymbolHandle

# load library
if (os.name=='posix'):
    path = os.path.abspath('libpass_lib.so')
    mx.library.load(path)
elif (os.name=='nt'):
    path = os.path.abspath('libpass_lib.dll')
    mx.library.load(path)


class _NCCLReduceHelper(object):
    _init = False
    nccl_id = None
    num_gpus = None
    rank = None

    @staticmethod
    def init(num_gpus, root_rank):
        """Communicate the NCCL unique id"""
        cls = _NCCLReduceHelper
        if not cls._init:
            cls._init = True
            import ctypes
            try:
                from mpi4py import MPI
            except:
                raise ImportError("Spatial parallel modules require mpi4py package.")
            import numpy as np
            nccl_id_size = ctypes.c_int()
            check_call(_LIB.MXNCCLGetUniqueIdSize(ctypes.byref(nccl_id_size)))
            nccl_id_size = nccl_id_size.value
            cls.nccl_id = np.zeros(nccl_id_size, np.byte)
            check_call(_LIB.MXNCCLGetUniqueId(
                cls.nccl_id.ctypes.data_as(ctypes.c_void_p)))
            global_comm = MPI.COMM_WORLD
            rank = global_comm.rank
            color = rank / num_gpus
            comm = global_comm.Split(color, rank)
            comm.Bcast([cls.nccl_id, nccl_id_size, MPI.BYTE], root=0)
            cls.num_gpus = num_gpus
            cls.rank = rank % num_gpus
            cls.root_rank = root_rank % num_gpus
        assert num_gpus == cls.num_gpus
###############################################
# Test with not consuming params
###############################################
# example model, ops do not have args (use outputs from other ops as inputs)
a = mx.sym.var('a')
b = mx.sym.var('b')
c = a + b
d = mx.sym.exp(c)
sym = mx.sym.log(d)


class Easynet(nn.HybridBlock):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Dense(in_units=2, units=2, flatten=False)
        #self.l2 = nn.Dense(in_units=2, units=2, flatten=False)
        #self.act1 = get_activation('relu')

        #self.seq.add(nn.Dense(in_units=2, units=2, flatten=False))
        #self.seq.add(get_activation('relu'))
        #self.seq.add(nn.Dense(in_units=2, units=2, flatten=False))
        #self.seq.register_op_hook(mon_callback,  monitor_all=True)
        #self.l1.register_op_hook(mon_callback,  monitor_all=True)


    def forward(self, input):
        input = self.l1(input)
        #print(input)
        #input = self.act1(input)
        return input


def test_model(pass_name):
    from mxnet import gluon
    from mxnet.gluon import Block, nn, HybridBlock
    from mxnet import init



    hvd.init()
    rank = hvd.rank()
    size = hvd.size()
    ctx = mx.gpu(rank)

    np.random.seed(1234 + 10 * rank)
    mx.random.seed(1234 + 10 * rank)

    num_gpus = size
    root_rank = 0
    helper = _NCCLReduceHelper
    helper.init(num_gpus, 0)
    kwargs = {
        'num_gpus': num_gpus,
        'rank': helper.rank,
        'root_rank': helper.root_rank,
        'nccl_unique_id': helper.nccl_id.ctypes.data}

    options = {"num_gpus":num_gpus, "rank": rank, "nccl_unique_id": helper.nccl_id.ctypes.data}

    model = Easynet()

    if rank == 0:
        model.l1.weight.initialize(init=init.One(), ctx=ctx)
        model.l1.bias.initialize(init=init.One(), ctx=ctx)
    else:
        model.l1.weight.initialize(init = init.Zero(), ctx = ctx)
        model.l1.bias.initialize(init=init.Zero(), ctx=ctx)
    model.hybridize()
    #param_dict = classify_net.collect_params()
    #params = [p for p in param_dict.values() if p.grad_req != 'null']
    params = model.collect_params()
    index = 0
    dic = {}
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
    cor_rank = trainer.correspond_ranks()

    for name in params:
        type = name.split('.')[-1]
        index = cor_rank[params[name]._uuid]
        new_name = params[name]._uuid.replace('-', '_') + '_' + type
        dic[new_name] = index

    options.update(dic)
    print(options)
    #return
    backward_options = {"partition_grad":True, "current_rank":rank}
    for k in options:
        if k in ['num_gpus', 'rank', 'nccl_unique_id']:
            continue
        backward_options['ncclreduce_' + k + '_backward'] = options[k]
    print(backward_options)
    x = np.ones((1,2), ctx = ctx)
    label = np.ones((2, ), ctx = ctx) * rank
    #print(options)

    loss_function = gluon.loss.L2Loss()

    model.optimize_for(x, backend = pass_name, **options)
    #model.export("my_reduce_" + str(rank))
    for i in range(1):
        with mx.autograd.record():
            out = model(x)
            loss = loss_function(out, label).mean() / size
            print("now call backward in python")
            #print(loss.backward)
            #print(loss)
            #print(loss.backward)
            loss.backward(backward_option = backward_options)
    mx.npx.waitall()
    mx.nd.waitall()
    for name in params:
        print(name, params[name].list_grad()[0])


    #print(out)


def test_reduce(pass_name):
    from mxnet import gluon
    from mxnet.gluon import Block, nn, HybridBlock
    from mxnet import init

    hvd.init()
    rank = hvd.rank()
    size = hvd.size()
    ctx = mx.gpu(rank)

    num_gpus = size
    root_rank = 0
    helper = _NCCLReduceHelper
    helper.init(num_gpus, root_rank)
    kwargs = {
        'num_gpus': num_gpus,
        'rank': helper.rank,
        'root_rank': helper.root_rank,
        'nccl_unique_id': helper.nccl_id.ctypes.data}

    options = {"rank": rank, "nccl_unique_id":helper.nccl_id.ctypes.data}

    a = mx.sym.var('a')
    sym = mx.sym.contrib.NCCLReduce(a, **kwargs)

    inputs = [a]
    sym_block = nn.SymbolBlock(sym, inputs)
    sym_block.initialize(ctx = ctx)
    p = mx.nd.ones((2, 3), ctx=ctx) * (rank + 2)
    sym_block.optimize_for(p, backend = pass_name, **options)
    sym_block.export("my_choice")
    return

    print(p)
    out = sym_block(p)
    mx.nd.waitall()
    mx.npx.waitall()
    print(out)
    return
    class ReduceNet(nn.HybridBlock):
        def __init__(self):
            super().__init__()
            self.reduce_layer = NCCLReduce(num_gpus=2, root_rank=0)
        def forward(self, x):
            x = self.reduce_layer(x)
            return x

    net = ReduceNet()
    net.initialize(ctx=ctx)
    net.hybridize()
    data = mx.nd.ones((3,2), ctx = ctx)
    net(data)
    #data = mx.np.ones((1, 10), ctx=ctx) * rank
    a = mx.sym.var('a')
    out = net(a)
    inputs = [a]
    out_sym = nn.SymbolBlock(out, inputs)
    out_sym.initialize(ctx = ctx)
    true_out = out_sym(mx.nd.ones((3,2)))

    print(true_out)


test_model('myPass')
