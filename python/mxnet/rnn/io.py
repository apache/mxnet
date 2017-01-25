# coding: utf-8
"""Definition of various recurrent neural network cells."""
from __future__ import print_function

import bisect
import random
import numpy as np

from ..io import DataIter, DataBatch
from .. import ndarray

def encode_sentences(sentences, vocab=None, invalid_label=-1, invalid_key='', start_label=0):
    idx = start_label
    if vocab is None:
        vocab = {invalid_key: invalid_label}
        new_vocab = True
    else:
        new_vocab = False
    res = []
    for sent in sentences:
        coded = []
        for word in sent:
            if word not in vocab:
                assert new_vocab, "Unknow token %s"%word
                if idx == invalid_label:
                    idx += 1
                vocab[word] = idx
                idx += 1
            coded.append(vocab[word])
        res.append(coded)

    return res, vocab

class BucketSentenceIter(DataIter):
    def __init__(self, sentences, batch_size, invalid_label=-1, dtype='float32',
                 buckets=None, data_name='data', label_name='softmax_label'):
        if not buckets:
            buckets = [i for i, j in enumerate(np.bincount([len(s) for s in sentences]))
                        if j >= batch_size]
        buckets.sort()

        ndiscard = 0
        self.data = [[] for _ in buckets]
        for i in xrange(len(sentences)):
            buck = bisect.bisect_left(buckets, len(sentences[i]))
            if buck == len(buckets):
                ndiscard += 1
                continue
            buff = np.full((buckets[buck],), invalid_label, dtype=dtype)
            buff[:len(sentences[i])] = sentences[i]
            self.data[buck].append(buff)

        print("WARNING: discarded %d sentences longer than the largest bucket."%ndiscard)

        self.default_bucket_key = max(buckets)

        self.provide_data = [(data_name, (batch_size, self.default_bucket_key))]
        self.provide_label = [(label_name, (batch_size, self.default_bucket_key))]

        self.batch_size = batch_size
        self.buckets = buckets
        self.data_name = data_name
        self.label_name = label_name
        self.dtype = dtype
        self.invalid_label = invalid_label

        self.idx = []
        for i, buck in enumerate(self.data):
            self.idx.extend([(i, j) for j in range(0, len(buck) - batch_size + 1, batch_size)])
        self.curr_idx = 0

    def reset(self):
        self.curr_idx = 0
        random.shuffle(self.idx)
        for buck in self.data:
            random.shuffle(buck)

    def next(self):
        if self.curr_idx == len(self.idx):
            raise StopIteration
        i, j = self.idx[self.curr_idx]
        self.curr_idx += 1

        data = np.asarray(self.data[i][j:j+self.batch_size], dtype=self.dtype)
        label = np.empty_like(data)
        label[:, :-1] = data[:, 1:]
        label[:, -1] = self.invalid_label


        return DataBatch([ndarray.array(data, dtype=self.dtype)],
                         [ndarray.array(label, dtype=self.dtype)],
                         bucket_key=self.buckets[i],
                         provide_data=[(self.data_name, data.shape)],
                         provide_label=[(self.label_name, label.shape)])




















