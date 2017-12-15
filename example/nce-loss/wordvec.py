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

# pylint: disable=invalid-name, missing-docstring, deprecated-module, len-as-condition
# pylint: disable=too-many-instance-attributes
from __future__ import print_function

import logging
import math
import random
from optparse import OptionParser

import mxnet as mx
from nce import nce_loss, NceAuc


def get_net(vocab_size, num_input, num_label_):
    data = mx.sym.Variable('data')
    label = mx.sym.Variable('label')
    label_weight = mx.sym.Variable('label_weight')
    embed_weight = mx.sym.Variable('embed_weight')
    data_embed = mx.sym.Embedding(data=data, input_dim=vocab_size,
                                  weight=embed_weight,
                                  output_dim=100, name='data_embed')
    datavec = mx.sym.SliceChannel(data=data_embed,
                                  num_outputs=num_input,
                                  squeeze_axis=1, name='data_slice')
    pred = datavec[0]
    for i in range(1, num_input):
        pred = pred + datavec[i]
    return nce_loss(data=pred,
                    label=label,
                    label_weight=label_weight,
                    embed_weight=embed_weight,
                    vocab_size=vocab_size,
                    num_hidden=100,
                    num_label=num_label_)


def load_data(name):
    buf = open(name).read()
    tks = buf.split(' ')
    vocab = {}
    freq = [0]
    data = []
    for tk in tks:
        if len(tk) == 0:
            continue
        if tk not in vocab:
            vocab[tk] = len(vocab) + 1
            freq.append(0)
        wid = vocab[tk]
        data.append(wid)
        freq[wid] += 1
    negative = []
    for i, v in enumerate(freq):
        if i == 0 or v < 5:
            continue
        v = int(math.pow(v * 1.0, 0.75))
        negative += [i for _ in range(v)]
    return data, negative, vocab, freq


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


class DataIter(mx.io.DataIter):
    def __init__(self, name, batch_size_, num_label_):
        super(DataIter, self).__init__()
        self.batch_size = batch_size_
        self.data, self.negative, self.vocab, self.freq = load_data(name)
        self.vocab_size = 1 + len(self.vocab)
        print("Vocabulary Size: {}".format(self.vocab_size))
        self.num_label = num_label_
        self.provide_data = [('data', (batch_size_, num_label_ - 1))]
        self.provide_label = [('label', (self.batch_size, num_label_)),
                              ('label_weight', (self.batch_size, num_label_))]

    def sample_ne(self):
        return self.negative[random.randint(0, len(self.negative) - 1)]

    def __iter__(self):
        batch_data = []
        batch_label = []
        batch_label_weight = []
        start = random.randint(0, self.num_label - 1)
        for i in range(start, len(self.data) - self.num_label - start, self.num_label):
            context = self.data[i: i + self.num_label // 2] \
                      + self.data[i + 1 + self.num_label // 2: i + self.num_label]
            target_word = self.data[i + self.num_label // 2]
            if self.freq[target_word] < 5:
                continue
            target = [target_word] + [self.sample_ne() for _ in range(self.num_label - 1)]
            target_weight = [1.0] + [0.0 for _ in range(self.num_label - 1)]
            batch_data.append(context)
            batch_label.append(target)
            batch_label_weight.append(target_weight)
            if len(batch_data) == self.batch_size:
                data_all = [mx.nd.array(batch_data)]
                label_all = [mx.nd.array(batch_label), mx.nd.array(batch_label_weight)]
                data_names = ['data']
                label_names = ['label', 'label_weight']
                batch_data = []
                batch_label = []
                batch_label_weight = []
                yield SimpleBatch(data_names, data_all, label_names, label_all)

    def reset(self):
        pass


if __name__ == '__main__':
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=head)

    parser = OptionParser()
    parser.add_option("-g", "--gpu", action="store_true", dest="gpu", default=False,
                      help="use gpu")
    options, args = parser.parse_args()

    batch_size = 256
    num_label = 5

    data_train = DataIter("./data/text8", batch_size, num_label)

    network = get_net(data_train.vocab_size, num_label - 1, num_label)

    devs = mx.cpu()
    if options.gpu:
        devs = mx.gpu()

    model = mx.mod.Module(
        symbol=network,
        data_names=[x[0] for x in data_train.provide_data],
        label_names=[y[0] for y in data_train.provide_label],
        context=[devs]
    )

    print("Training on {}".format("GPU" if options.gpu else "CPU"))
    metric = NceAuc()
    model.fit(
        train_data=data_train,
        num_epoch=20,
        optimizer='sgd',
        optimizer_params={'learning_rate': 0.3, 'momentum': 0.9, 'wd': 0.0000},
        initializer=mx.init.Xavier(factor_type='in', magnitude=2.34),
        eval_metric=metric,
        batch_end_callback=mx.callback.Speedometer(batch_size, 50)
    )
