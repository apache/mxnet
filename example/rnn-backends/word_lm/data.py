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

import os, gzip
import sys
import mxnet as mx
import numpy as np

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

class Corpus(object):
    def __init__(self, path, name):
        self.dictionary = Dictionary()
        if name == 'ptb':
            self.train = self.tokenize(path + 'ptb/ptb.train.txt')
            self.valid = self.tokenize(path + 'ptb/ptb.valid.txt')
            self.test  = self.tokenize(path + 'ptb/ptb.test.txt')
        elif name == 'wikitext-2':
            self.train = self.tokenize(path + 'wikitext-2/wiki.train.tokens')
            self.valid = self.tokenize(path + 'wikitext-2/wiki.valid.tokens')
            self.test  = self.tokenize(path + 'wikitext-2/wiki.test.tokens')
        else:
            assert 0, "Invalid dataset name %s. " \
                      "Valid ones are ptb/wikitext-2." % name

    def tokenize(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Add words to the dictionary
        with open(path, 'r') as f:
            tokens = 0
            for line in f:
                words = line.split() + ['<eos>']
                tokens += len(words)
                for word in words:
                    self.dictionary.add_word(word)

        # Tokenize file content
        with open(path, 'r') as f:
            ids = np.zeros((tokens,), dtype='int32')
            token = 0
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    ids[token] = self.dictionary.word2idx[word]
                    token += 1

        return mx.nd.array(ids, dtype='int32')

def batchify(data, batch_size, layout):
    """Reshape data into (batch_size, num_examples)"""
    nbatch = data.shape[0] // batch_size
    data = data[:nbatch*batch_size]

    if layout == 'NT':
        data = data.reshape((batch_size, nbatch))
    else:
        data = data.reshape((batch_size, nbatch)).T

    return data

class CorpusIter(mx.io.DataIter):
    "An iterator that returns the a batch of sequence each time"
    def __init__(self, source, batch_size, bptt, layout='NT'):
        super(CorpusIter, self).__init__()
        self._batch_size = batch_size
        self._bptt = bptt
        self._layout = layout

        if layout == 'NT':
            self.provide_data  = [mx.io.DataDesc(name='data' , shape=(batch_size, bptt),
                                                 dtype=np.  int32, layout=layout)]
            self.provide_label = [mx.io.DataDesc(name='label', shape=(batch_size, bptt),
                                                 dtype=np.float32, layout=layout)]
        elif layout == 'TN': # CuDNN implementation expects time-major layout.
            self.provide_data  = [mx.io.DataDesc(name='data' , shape=(bptt, batch_size),
                                                 dtype=np.  int32, layout=layout)]
            self.provide_label = [mx.io.DataDesc(name='label', shape=(bptt, batch_size),
                                                 dtype=np.float32, layout=layout)]
        else:
            assert 0, "Invalid data layout argument. Valid ones are NT/TN."

        self._index = 0
        self._source = batchify(source, batch_size, layout)

    def iter_next(self):
        layout = self._layout
        i = self._index
        if i+self._bptt > self._source.shape[1 if layout == 'NT' else 0] - 1:
            return False

        if layout == 'NT':
            self._next_data  = self._source[:,i  :i  +self._bptt].astype(np.  int32)
            self._next_label = self._source[:,i+1:i+1+self._bptt].astype(np.float32)
        else:
            self._next_data  = self._source[i  :i  +self._bptt].astype(np.  int32)
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
        self._next_data  = None
        self._next_label = None

    def getdata(self):
        return [self._next_data]

    def getlabel(self):
        return [self._next_label]

