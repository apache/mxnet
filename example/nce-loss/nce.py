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

from operator import itemgetter

import mxnet as mx
import numpy as np


def nce_loss(data, label, label_weight, embed_weight, vocab_size, num_hidden):
    label_embed = mx.sym.Embedding(data=label, input_dim=vocab_size,
                                   weight=embed_weight,
                                   output_dim=num_hidden, name='label_embed')
    data = mx.sym.Reshape(data=data, shape=(-1, 1, num_hidden))
    pred = mx.sym.broadcast_mul(data, label_embed)
    pred = mx.sym.sum(data=pred, axis=2)
    return mx.sym.LogisticRegressionOutput(data=pred,
                                           label=label_weight)


def nce_loss_subwords(
        data, label, label_mask, label_weight, embed_weight, vocab_size, num_hidden):
    """NCE-Loss layer under subword-units input.
    """
    # get subword-units embedding.
    label_units_embed = mx.sym.Embedding(data=label,
                                         input_dim=vocab_size,
                                         weight=embed_weight,
                                         output_dim=num_hidden)
    # get valid subword-units embedding with the help of label_mask
    # it's achieved by multiplying zeros to useless units in order to handle variable-length input.
    label_units_embed = mx.sym.broadcast_mul(lhs=label_units_embed,
                                             rhs=label_mask,
                                             name='label_units_embed')
    # sum over them to get label word embedding.
    label_embed = mx.sym.sum(label_units_embed, axis=2, name='label_embed')

    # by boardcast_mul and sum you can get prediction scores in all label_embed inputs,
    # which is easy to feed into LogisticRegressionOutput and make your code more concise.
    data = mx.sym.Reshape(data=data, shape=(-1, 1, num_hidden))
    pred = mx.sym.broadcast_mul(data, label_embed)
    pred = mx.sym.sum(data=pred, axis=2)

    return mx.sym.LogisticRegressionOutput(data=pred,
                                           label=label_weight)


class NceAccuracy(mx.metric.EvalMetric):
    def __init__(self):
        super(NceAccuracy, self).__init__('nce-accuracy')

    def update(self, labels, preds):
        label_weight = labels[1].asnumpy()
        preds = preds[0].asnumpy()
        for i in range(preds.shape[0]):
            if np.argmax(label_weight[i]) == np.argmax(preds[i]):
                self.sum_metric += 1
            self.num_inst += 1


class NceAuc(mx.metric.EvalMetric):
    def __init__(self):
        super(NceAuc, self).__init__('nce-auc')

    def update(self, labels, preds):
        label_weight = labels[1].asnumpy()
        preds = preds[0].asnumpy()
        tmp = []
        for i in range(preds.shape[0]):
            for j in range(preds.shape[1]):
                tmp.append((label_weight[i][j], preds[i][j]))
        tmp = sorted(tmp, key=itemgetter(1), reverse=True)
        m = 0.0
        n = 0.0
        z = 0.0
        k = 0
        for a, _ in tmp:
            if a > 0.5:
                m += 1.0
                z += len(tmp) - k
            else:
                n += 1.0
            k += 1
        z -= m * (m + 1.0) / 2.0
        z /= m
        z /= n
        self.sum_metric += z
        self.num_inst += 1


class NceLSTMAuc(mx.metric.EvalMetric):
    def __init__(self):
        super(NceLSTMAuc, self).__init__('nce-lstm-auc')

    def update(self, labels, preds):
        preds = np.array([x.asnumpy() for x in preds])
        preds = preds.reshape((preds.shape[0] * preds.shape[1], preds.shape[2]))
        label_weight = labels[1].asnumpy()
        label_weight = label_weight.transpose((1, 0, 2))
        label_weight = label_weight.reshape((preds.shape[0], preds.shape[1]))

        tmp = []
        for i in range(preds.shape[0]):
            for j in range(preds.shape[1]):
                tmp.append((label_weight[i][j], preds[i][j]))
        tmp = sorted(tmp, key=itemgetter(1), reverse=True)
        m = 0.0
        n = 0.0
        z = 0.0
        k = 0
        for a, _ in tmp:
            if a > 0.5:
                m += 1.0
                z += len(tmp) - k
            else:
                n += 1.0
            k += 1
        z -= m * (m + 1.0) / 2.0
        z /= m
        z /= n
        self.sum_metric += z
        self.num_inst += 1
