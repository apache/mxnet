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

# pylint: disable=missing-docstring
from __future__ import print_function

import numpy as np
import mxnet as mx
try:
    import cPickle as pickle
except ImportError:
    import pickle


def extract_feature(sym, args, auxs, data_iter, N, xpu=mx.cpu()):
    input_buffs = [mx.nd.empty(shape, ctx=xpu) for k, shape in data_iter.provide_data]
    input_names = [k for k, shape in data_iter.provide_data]
    args = dict(args, **dict(zip(input_names, input_buffs)))
    exe = sym.bind(xpu, args=args, aux_states=auxs)
    outputs = [[] for _ in exe.outputs]
    output_buffs = None

    data_iter.hard_reset()
    for batch in data_iter:
        for data, buff in zip(batch.data, input_buffs):
            data.copyto(buff)
        exe.forward(is_train=False)
        if output_buffs is None:
            output_buffs = [mx.nd.empty(i.shape, ctx=mx.cpu()) for i in exe.outputs]
        else:
            for out, buff in zip(outputs, output_buffs):
                out.append(buff.asnumpy())
        for out, buff in zip(exe.outputs, output_buffs):
            out.copyto(buff)
    for out, buff in zip(outputs, output_buffs):
        out.append(buff.asnumpy())
    outputs = [np.concatenate(i, axis=0)[:N] for i in outputs]
    return dict(zip(sym.list_outputs(), outputs))


class MXModel(object):
    def __init__(self, xpu=mx.cpu(), *args, **kwargs):
        self.xpu = xpu
        self.loss = None
        self.args = {}
        self.args_grad = {}
        self.args_mult = {}
        self.auxs = {}
        self.setup(*args, **kwargs)

    def save(self, fname):
        args_save = {key: v.asnumpy() for key, v in self.args.items()}
        with open(fname, 'wb') as fout:
            pickle.dump(args_save, fout)

    def load(self, fname):
        with open(fname, 'rb') as fin:
            args_save = pickle.load(fin)
            for key, v in args_save.items():
                if key in self.args:
                    self.args[key][:] = v

    def setup(self, *args, **kwargs):
        raise NotImplementedError("must override this")
