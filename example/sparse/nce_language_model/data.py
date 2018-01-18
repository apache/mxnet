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

import os, gzip, sys
import mxnet as mx
import numpy as np
import data_utils

class DummyIter(mx.io.DataIter):
    "A dummy iterator that always returns the same batch, used for speed testing"
    def __init__(self, real_iter):
        super(DummyIter, self).__init__()
        self.real_iter = real_iter
        self.provide_data = real_iter.provide_data
        self.provide_label = real_iter.provide_label
        self.batch_size = real_iter.batch_size

        for batch in real_iter:
            self.the_batch = batch
            break

    def __iter__(self):
        return self

    def next(self):
        return self.the_batch

class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []
        self.word_count = []

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
            self.word_count.append(0)
        index = self.word2idx[word]
        self.word_count[index] += 1
        return index

    def __len__(self):
        return len(self.idx2word)

    def unigram(self):
        prob = mx.nd.array(self.word_count)
        total_count = mx.nd.sum(prob)
        return prob / total_count

    def save(self, path, replace_unk=True):
        f = open(path, "w")
        for idx, word in enumerate(self.idx2word):
            count = self.word_count[idx]
            if replace_unk and word == '<unk>':
                word = '<UNK>'
            f.write("%s %d\n" %(word, count))
        f.close()

class Corpus(object):
    def __init__(self, path, prepend=False):
        self.dictionary = Dictionary()
        self.train = self.tokenize(path + 'train.txt', prepend)
        self.valid = self.tokenize(path + 'valid.txt', prepend)
        self.test = self.tokenize(path + 'test.txt', prepend)

    def tokenize(self, path, prepend):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Add words to the dictionary
        with open(path, 'r') as f:
            tokens = 0
            for line in f:
                words = ['<S>'] + line.split() + ['<S>'] if prepend else line.split() + ['<S>']
                tokens += len(words)
                for word in words:
                    self.dictionary.add_word(word)

        # Tokenize file content
        with open(path, 'r') as f:
            ids = np.zeros((tokens,), dtype='int32')
            token = 0
            for line in f:
                words = ['<S>'] + line.split() + ['<S>'] if prepend else line.split() + ['<S>']
                for word in words:
                    ids[token] = self.dictionary.word2idx[word]
                    token += 1

        return mx.nd.array(ids, dtype='int32')

def batchify(data, batch_size):
    """Reshape data into (num_example, batch_size)"""
    nbatch = data.shape[0] // batch_size
    data = data[:nbatch * batch_size]
    data = data.reshape((batch_size, nbatch)).T
    return data

class CorpusIter(mx.io.DataIter):
    "An iterator that returns the a batch of sequence each time"
    def __init__(self, source, batch_size, bptt):
        super(CorpusIter, self).__init__()
        self.batch_size = batch_size
        self.provide_data = [('data', (bptt, batch_size), np.int32)]
        self.provide_label = [('label', (bptt, batch_size))]
        self._index = 0
        self._bptt = bptt
        self._source = batchify(source, batch_size)

    def iter_next(self):
        i = self._index
        if i+self._bptt > self._source.shape[0] - 1:
            return False
        self._next_data = self._source[i:i+self._bptt]
        self._next_label = self._source[i+1:i+1+self._bptt].astype(np.float32)
        self._index += self._bptt
        return True

    def next(self):
        if self.iter_next():
            return mx.io.DataBatch(data=self.getdata(), label=self.getlabel())
        else:
            raise StopIteration

    def reset(self):
        self._index = 0
        self._next_data = None
        self._next_label = None

    def getdata(self):
        return [self._next_data]

    def getlabel(self):
        return [self._next_label]

class MultiSentenceIter(mx.io.DataIter):
    "An iterator that returns the a batch of sequence each time"
    def __init__(self, data_file, vocab, batch_size, bptt):
        super(MultiSentenceIter, self).__init__()
        self.batch_size = batch_size
        self.bptt = bptt
        self.provide_data = [('data', (batch_size, bptt), np.int32), ('mask', (batch_size, bptt))]
        self.provide_label = [('label', (batch_size, bptt))]
        self.vocab = vocab
        self.data_file = data_file
        self._dataset = data_utils.Dataset(self.vocab, data_file, deterministic=True)
        self._iter = self._dataset.iterate_once(batch_size, bptt)

    def iter_next(self):
        data = self._iter.next()
        if data is None:
            return False
        self._next_data = mx.nd.array(data[0], dtype=np.int32)
        self._next_label = mx.nd.array(data[1])
        self._next_mask = mx.nd.array(data[2])
        self._next_mask[:] = 1
        return True

    def next(self):
        if self.iter_next():
            return mx.io.DataBatch(data=self.getdata(), label=self.getlabel())
        else:
            raise StopIteration

    def reset(self):
        print('reset')
        self._dataset = data_utils.Dataset(self.vocab, self.data_file, deterministic=True)
        self._iter = self._dataset.iterate_once(self.batch_size, self.bptt)
        self._next_data = None
        self._next_label = None
        self._next_mask = None

    def getdata(self):
        return [self._next_data, self._next_mask]

    def getlabel(self):
        return [self._next_label]
