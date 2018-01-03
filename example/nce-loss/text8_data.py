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

from collections import Counter
import logging
import math
import random

import mxnet as mx
import numpy as np


def _load_data(name):
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


class SubwordData(object):
    def __init__(self, data, units, weights, negative_units, negative_weights, vocab, units_vocab,
                 freq, max_len):
        self.data = data
        self.units = units
        self.weights = weights
        self.negative_units = negative_units
        self.negative_weights = negative_weights
        self.vocab = vocab
        self.units_vocab = units_vocab
        self.freq = freq
        self.max_len = max_len


def _get_subword_units(token, gram):
    """Return subword-units presentation, given a word/token.
    """
    if token == '</s>':  # special token for padding purpose.
        return [token]
    t = '#' + token + '#'
    return [t[i:i + gram] for i in range(0, len(t) - gram + 1)]


def _get_subword_representation(wid, vocab_inv, units_vocab, max_len, gram, padding_char):
    token = vocab_inv[wid]
    units = [units_vocab[unit] for unit in _get_subword_units(token, gram)]
    weights = [1] * len(units) + [0] * (max_len - len(units))
    units = units + [units_vocab[padding_char]] * (max_len - len(units))
    return units, weights


def _prepare_subword_units(tks, gram, padding_char):
    # statistics on units
    units_vocab = {padding_char: 1}
    max_len = 0
    unit_set = set()
    logging.info('grams: %d', gram)
    logging.info('counting max len...')
    for tk in tks:
        res = _get_subword_units(tk, gram)
        unit_set.update(i for i in res)
        if max_len < len(res):
            max_len = len(res)
    logging.info('preparing units vocab...')
    for unit in unit_set:
        if len(unit) == 0:
            continue
        if unit not in units_vocab:
            units_vocab[unit] = len(units_vocab)
        # uid = units_vocab[unit]
    return units_vocab, max_len


def _load_data_as_subword_units(name, min_count, gram, max_subwords, padding_char):
    tks = []
    fread = open(name, 'rb')
    logging.info('reading corpus from file...')
    for line in fread:
        line = line.strip().decode('utf-8')
        tks.extend(line.split(' '))

    logging.info('Total tokens: %d', len(tks))

    tks = [tk for tk in tks if len(tk) <= max_subwords]
    c = Counter(tks)

    logging.info('Total vocab: %d', len(c))

    vocab = {}
    vocab_inv = {}
    freq = [0]
    data = []
    for tk in tks:
        if len(tk) == 0:
            continue
        if tk not in vocab:
            vocab[tk] = len(vocab)
            freq.append(0)
        wid = vocab[tk]
        vocab_inv[wid] = tk
        data.append(wid)
        freq[wid] += 1

    negative = []
    for i, v in enumerate(freq):
        if i == 0 or v < min_count:
            continue
        v = int(math.pow(v * 1.0, 0.75))  # sample negative w.r.t. its frequency
        negative += [i for _ in range(v)]

    logging.info('counting subword units...')
    units_vocab, max_len = _prepare_subword_units(tks, gram, padding_char)
    logging.info('vocabulary size: %d', len(vocab))
    logging.info('subword unit size: %d', len(units_vocab))

    logging.info('generating input data...')
    units = []
    weights = []
    for wid in data:
        word_units, weight = _get_subword_representation(
            wid, vocab_inv, units_vocab, max_len, gram, padding_char)
        units.append(word_units)
        weights.append(weight)

    negative_units = []
    negative_weights = []
    for wid in negative:
        word_units, weight = _get_subword_representation(
            wid, vocab_inv, units_vocab, max_len, gram, padding_char)
        negative_units.append(word_units)
        negative_weights.append(weight)

    return SubwordData(
        data=data, units=units, weights=weights, negative_units=negative_units,
        negative_weights=negative_weights, vocab=vocab, units_vocab=units_vocab,
        freq=freq, max_len=max_len
    )


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


class DataIterWords(mx.io.DataIter):
    def __init__(self, name, batch_size, num_label):
        super(DataIterWords, self).__init__()
        self.batch_size = batch_size
        self.data, self.negative, self.vocab, self.freq = _load_data(name)
        self.vocab_size = 1 + len(self.vocab)
        print("Vocabulary Size: {}".format(self.vocab_size))
        self.num_label = num_label
        self.provide_data = [('data', (batch_size, num_label - 1))]
        self.provide_label = [('label', (self.batch_size, num_label)),
                              ('label_weight', (self.batch_size, num_label))]

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


class DataIterLstm(mx.io.DataIter):
    def __init__(self, name, batch_size, seq_len, num_label, init_states):
        super(DataIterLstm, self).__init__()
        self.batch_size = batch_size
        self.data, self.negative, self.vocab, self.freq = _load_data(name)
        self.vocab_size = 1 + len(self.vocab)
        print("Vocabulary Size: {}".format(self.vocab_size))
        self.seq_len = seq_len
        self.num_label = num_label
        self.init_states = init_states
        self.init_state_names = [x[0] for x in self.init_states]
        self.init_state_arrays = [mx.nd.zeros(x[1]) for x in init_states]
        self.provide_data = [('data', (batch_size, seq_len))] + init_states
        self.provide_label = [('label', (self.batch_size, seq_len, num_label)),
                              ('label_weight', (self.batch_size, seq_len, num_label))]

    def sample_ne(self):
        return self.negative[random.randint(0, len(self.negative) - 1)]

    def __iter__(self):
        batch_data = []
        batch_label = []
        batch_label_weight = []
        for i in range(0, len(self.data) - self.seq_len - 1, self.seq_len):
            data = self.data[i: i+self.seq_len]
            label = [[self.data[i+k+1]] \
                     + [self.sample_ne() for _ in range(self.num_label-1)]\
                     for k in range(self.seq_len)]
            label_weight = [[1.0] \
                            + [0.0 for _ in range(self.num_label-1)]\
                            for k in range(self.seq_len)]

            batch_data.append(data)
            batch_label.append(label)
            batch_label_weight.append(label_weight)
            if len(batch_data) == self.batch_size:
                data_all = [mx.nd.array(batch_data)] + self.init_state_arrays
                label_all = [mx.nd.array(batch_label), mx.nd.array(batch_label_weight)]
                data_names = ['data'] + self.init_state_names
                label_names = ['label', 'label_weight']
                batch_data = []
                batch_label = []
                batch_label_weight = []
                yield SimpleBatch(data_names, data_all, label_names, label_all)

    def reset(self):
        pass


class DataIterSubWords(mx.io.DataIter):
    def __init__(self, fname, batch_size, num_label, min_count, gram, max_subwords, padding_char):
        super(DataIterSubWords, self).__init__()
        self.batch_size = batch_size
        self.min_count = min_count
        self.swd = _load_data_as_subword_units(
            fname,
            min_count=min_count,
            gram=gram,
            max_subwords=max_subwords,
            padding_char=padding_char)
        self.vocab_size = len(self.swd.units_vocab)
        self.num_label = num_label
        self.provide_data = [('data', (batch_size, num_label - 1, self.swd.max_len)),
                             ('mask', (batch_size, num_label - 1, self.swd.max_len, 1))]
        self.provide_label = [('label', (self.batch_size, num_label, self.swd.max_len)),
                              ('label_weight', (self.batch_size, num_label)),
                              ('label_mask', (self.batch_size, num_label, self.swd.max_len, 1))]

    def sample_ne(self):
        # a negative sample.
        return self.swd.negative_units[random.randint(0, len(self.swd.negative_units) - 1)]

    def sample_ne_indices(self):
        return [random.randint(0, len(self.swd.negative_units) - 1)
                for _ in range(self.num_label - 1)]

    def __iter__(self):
        logging.info('DataIter start.')
        batch_data = []
        batch_data_mask = []
        batch_label = []
        batch_label_mask = []
        batch_label_weight = []
        start = random.randint(0, self.num_label - 1)
        for i in range(start, len(self.swd.units) - self.num_label - start, self.num_label):
            context_units = self.swd.units[i: i + self.num_label // 2] + \
                            self.swd.units[i + 1 + self.num_label // 2: i + self.num_label]
            context_mask = self.swd.weights[i: i + self.num_label // 2] + \
                           self.swd.weights[i + 1 + self.num_label // 2: i + self.num_label]
            target_units = self.swd.units[i + self.num_label // 2]
            target_word = self.swd.data[i + self.num_label // 2]
            if self.swd.freq[target_word] < self.min_count:
                continue
            indices = self.sample_ne_indices()
            target = [target_units] + [self.swd.negative_units[i] for i in indices]
            target_weight = [1.0] + [0.0 for _ in range(self.num_label - 1)]
            target_mask = [self.swd.weights[i + self.num_label // 2]] +\
                          [self.swd.negative_weights[i] for i in indices]

            batch_data.append(context_units)
            batch_data_mask.append(context_mask)
            batch_label.append(target)
            batch_label_mask.append(target_mask)
            batch_label_weight.append(target_weight)

            if len(batch_data) == self.batch_size:
                # reshape for broadcast_mul
                batch_data_mask = np.reshape(
                    batch_data_mask, (self.batch_size, self.num_label - 1, self.swd.max_len, 1))
                batch_label_mask = np.reshape(
                    batch_label_mask, (self.batch_size, self.num_label, self.swd.max_len, 1))
                data_all = [mx.nd.array(batch_data), mx.nd.array(batch_data_mask)]
                label_all = [
                    mx.nd.array(batch_label),
                    mx.nd.array(batch_label_weight),
                    mx.nd.array(batch_label_mask)
                ]
                data_names = ['data', 'mask']
                label_names = ['label', 'label_weight', 'label_mask']
                # clean up
                batch_data = []
                batch_data_mask = []
                batch_label = []
                batch_label_weight = []
                batch_label_mask = []
                yield SimpleBatch(data_names, data_all, label_names, label_all)

    def reset(self):
        pass
