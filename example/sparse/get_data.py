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

def get_libsvm_data(data_dir, data_name, url):
    if not os.path.isdir(data_dir):
        os.mkdir(data_dir)
    os.chdir(data_dir)
    if (not os.path.exists(data_name)):
        print("Dataset " + data_name + " not present. Downloading now ...")
        import urllib
        zippath = os.path.join(data_dir, data_name + ".bz2")
        urllib.urlretrieve(url + data_name + ".bz2", zippath)
        os.system("bzip2 -d %r" % data_name + ".bz2")
        print("Dataset " + data_name + " is now present.")
    os.chdir("..")

def get_movielens_data(prefix):
    if not os.path.exists("%s.zip" % prefix):
        print("Dataset MovieLens 10M not present. Downloading now ...")
        os.system("wget http://files.grouplens.org/datasets/movielens/%s.zip" % prefix)
        os.system("unzip %s.zip" % prefix)
        os.system("cd ml-10M100K; sh split_ratings.sh; cd -;")

def get_movielens_iter(filename, batch_size, dummy_iter):
    """Not particularly fast code to parse the text file and load into NDArrays.
    return two data iters, one for train, the other for validation.
    """
    print("Preparing data iterators for " + filename + " ... ")
    user = []
    item = []
    score = []
    with open(filename, 'r') as f:
        num_samples = 0
        for line in f:
            tks = line.strip().split('::')
            if len(tks) != 4:
                continue
            num_samples += 1
            user.append((tks[0]))
            item.append((tks[1]))
            score.append((tks[2]))
            if dummy_iter and num_samples > batch_size * 10:
                break
    # convert to ndarrays
    user = mx.nd.array(user, dtype='int32')
    item = mx.nd.array(item)
    score = mx.nd.array(score)
    # prepare data iters
    data_train = {'user':user, 'item':item}
    label_train = {'score':score}
    iter_train = mx.io.NDArrayIter(data=data_train,label=label_train,
                                   batch_size=batch_size, shuffle=True)
    iter_train = DummyIter(iter_train) if dummy_iter else iter_train
    return mx.io.PrefetchingIter(iter_train)

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


class Corpus(object):
    def __init__(self, path):
        self.dictionary = Dictionary()
        self.train = self.tokenize(path + 'train.txt')
        self.valid = self.tokenize(path + 'valid.txt')
        self.test = self.tokenize(path + 'test.txt')

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

def batchify(data, batch_size):
    """Reshape data into (num_example, batch_size)"""
    nbatch = data.shape[0] // batch_size
    data = data[:nbatch * batch_size]
    data = data.reshape((batch_size, nbatch)).T
    return data

class CorpusIter(mx.io.DataIter):
    "An iterator that returns the a batch of sequence each time"
    def __init__(self, source, batch_size, bptt, k, unigram):
        super(CorpusIter, self).__init__()
        self.batch_size = batch_size
        self.provide_data = [('data', (batch_size, bptt))]
        self.provide_label = [('label', (batch_size, bptt))]
        self._index = 0
        self._bptt = bptt
        self._source = batchify(source, batch_size)
        self._unigram = unigram
        self._k = k

    def iter_next(self):
        i = self._index
        if i+self._bptt > self._source.shape[0] - 1:
            return False
        self._next_data = self._source[i:i+self._bptt]
        self._next_label = self._source[i+1:i+1+self._bptt]
        self._next_sample = mx.nd.random.multinomial(self._unigram, shape=(self._k,))
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
        return [self._next_data.T, self._next_sample]

    def getlabel(self):
        return [self._next_label.T]
