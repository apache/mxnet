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

from __future__ import print_function
from six.moves import range

import argparse
import subprocess
from itertools import product
from time import time

import mxnet as mx
import numpy as np
from mxnet import gluon


_parser = argparse.ArgumentParser(description='Benchmark foreach and while_loop on RNN tasks.')
_parser.add_argument('--benchmark', choices=["foreach", "while_loop"], required=True)
_parser.add_argument('--warmup_rounds', type=int, default=20)
_parser.add_argument('--test_rounds', type=int, default=100)
_parser.add_argument('--gpu', type=bool, default=False)
args = _parser.parse_args()


class ForeachRNN(gluon.HybridBlock):
    def __init__(self, cell, length, prefix=None, params=None):
        super(ForeachRNN, self).__init__(prefix=prefix, params=params)
        self.length = length
        self.cell = cell

    def hybrid_forward(self, F, inputs, states):
        out, states = F.contrib.foreach(self.cell, inputs, states)
        return out


class WhileRNN(gluon.HybridBlock):
    def __init__(self, cell, length, prefix=None, params=None):
        super(WhileRNN, self).__init__(prefix=prefix, params=params)
        self.length = length
        self.cell = cell

    def hybrid_forward(self, F, inputs, states):
        def _func(*states):
            i = states[0]
            s = states[1: ]
            data = inputs.take(i).squeeze(axis=0)
            out, new_s = self.cell(data, s)
            new_s = [i + 1] + new_s
            return out, new_s
        out, states = F.contrib.while_loop(
            cond=lambda i, *_: i < self.length,
            func=_func,
            loop_vars=states,
            max_iterations=self.length,
        )
        return out


def _zeros(shape, ctx):
    return mx.nd.zeros(shape=shape, ctx=ctx)


def _array(shape, ctx):
    return mx.nd.normal(loc=0.0, scale=1.0, shape=shape, ctx=ctx)


def _get_gpus():
    return range(mx.util.get_gpu_count())

def run_benchmark(cell_type, ctx, seq_len, batch_size, hidden_dim):
    obj = {"foreach": ForeachRNN, "while_loop": WhileRNN}[args.benchmark]
    inputs = _array((seq_len, batch_size, hidden_dim), ctx)
    states = [_array((batch_size, hidden_dim), ctx) for _ in cell_type(0).state_info()]
    if args.benchmark == "while_loop":
        states.insert(0, _zeros((1, ), ctx))

    for is_train, is_hyb_cell, is_hyb_layer in product([True, False], [False, True], [False, True]):
        cell = cell_type(hidden_dim)
        if is_hyb_cell:
            cell.hybridize(static_alloc=True)
        layer = obj(cell, seq_len)
        layer.initialize(ctx=ctx)
        if is_hyb_layer:
            layer.hybridize(static_alloc=True)
        print("is_train = %r, hybridize_cell = %r, hybridize_layer = %r" % (is_train, is_hyb_cell, is_hyb_layer))
        times = []
        for _ in range(args.warmup_rounds + args.test_rounds):
            tick = time()
            if not is_train:
                res = layer(inputs, states)
            else:
                with mx.autograd.record():
                    res = layer(inputs, states)
            if is_train:
                res.backward()
            mx.nd.waitall()
            tock = time()
            times.append((tock - tick) * 1000.0)
        times = times[args.warmup_rounds: ]
        print("Time used: mean = %.3f ms, std = %.3f ms" % (np.mean(times), np.std(times)))


def main():
    # testing configurations
    cell_types = [gluon.rnn.RNNCell,
                  gluon.rnn.GRUCell,
                  gluon.rnn.LSTMCell]
    ctxs = [mx.cpu(0)]
    if args.gpu:
        ctxs = ctxs + [mx.gpu(i) for i in _get_gpus()]
    seq_lens = [100]
    batch_sizes = [1, 32]
    hidden_dims = [512]
    print("--------------------------------------")
    print("Benchmarking", args.benchmark)
    for cell_type, ctx, seq_len, batch_size, hidden_dim in product(  \
        cell_types, ctxs, seq_lens, batch_sizes, hidden_dims):
        print("--------------------------------------")
        print("cell: %s  ctx: %s  length: %d  batch size: %d dim: %d" % \
              (cell_type.__name__, str(ctx), seq_len, batch_size, hidden_dim))
        run_benchmark(cell_type, ctx, seq_len, batch_size, hidden_dim)


if __name__ == "__main__":
    main()
