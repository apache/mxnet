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

import random

import mxnet as mx
import numpy as np


class SimpleBatch(object):
    def __init__(self, data_names, data, label_names, label):
        self.data = data
        self.label = label
        self.data_names = data_names
        self.label_names = label_names

    @property
    def provide_data(self):
        return [(n, x.shape) for n, x in zip(self.data_names, self.data)]

    @property
    def provide_label(self):
        return [(n, x.shape) for n, x in zip(self.label_names, self.label)]


class DataIterSoftmax(mx.io.DataIter):
    def __init__(self, count, batch_size, vocab_size, num_label, feature_size):
        super(DataIterSoftmax, self).__init__()
        self.batch_size = batch_size
        self.count = count
        self.vocab_size = vocab_size
        self.num_label = num_label
        self.feature_size = feature_size
        self.provide_data = [('data', (batch_size, feature_size))]
        self.provide_label = [('label', (self.batch_size,))]

    def mock_sample(self):
        ret = np.zeros(self.feature_size)
        rn = set()
        while len(rn) < 3:
            rn.add(random.randint(0, self.feature_size - 1))
        s = 0
        for k in rn:
            ret[k] = 1.0
            s *= self.feature_size
            s += k
        return ret, s % self.vocab_size

    def __iter__(self):
        for _ in range(self.count // self.batch_size):
            data = []
            label = []
            for _ in range(self.batch_size):
                d, l = self.mock_sample()
                data.append(d)
                label.append(l)
            data_all = [mx.nd.array(data)]
            label_all = [mx.nd.array(label)]
            data_names = ['data']
            label_names = ['label']
            yield SimpleBatch(data_names, data_all, label_names, label_all)

    def reset(self):
        pass


class DataIterNce(mx.io.DataIter):
    def __init__(self, count, batch_size, vocab_size, num_label, feature_size):
        super(DataIterNce, self).__init__()
        self.batch_size = batch_size
        self.count = count
        self.vocab_size = vocab_size
        self.num_label = num_label
        self.feature_size = feature_size
        self.provide_data = [('data', (batch_size, feature_size))]
        self.provide_label = [('label', (self.batch_size, num_label)),
                              ('label_weight', (self.batch_size, num_label))]

    def mock_sample(self):
        ret = np.zeros(self.feature_size)
        rn = set()
        while len(rn) < 3:
            rn.add(random.randint(0, self.feature_size - 1))
        s = 0
        for k in rn:
            ret[k] = 1.0
            s *= self.feature_size
            s += k
        la = [s % self.vocab_size] +\
             [random.randint(0, self.vocab_size - 1) for _ in range(self.num_label - 1)]
        return ret, la

    def __iter__(self):
        for _ in range(self.count // self.batch_size):
            data = []
            label = []
            label_weight = []
            for _ in range(self.batch_size):
                d, l = self.mock_sample()
                data.append(d)
                label.append(l)
                label_weight.append([1.0] + [0.0 for _ in range(self.num_label - 1)])
            data_all = [mx.nd.array(data)]
            label_all = [mx.nd.array(label), mx.nd.array(label_weight)]
            data_names = ['data']
            label_names = ['label', 'label_weight']
            yield SimpleBatch(data_names, data_all, label_names, label_all)

    def reset(self):
        pass
