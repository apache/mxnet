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

# pylint: disable=C0111,too-many-arguments,too-many-instance-attributes,too-many-locals,redefined-outer-name,fixme
# pylint: disable=superfluous-parens, no-member, invalid-name
from __future__ import print_function
import sys
import numpy as np
import mxnet as mx

# The interface of a data iter that works for bucketing
#
# DataIter
#   - default_bucket_key: the bucket key for the default symbol.
#
# DataBatch
#   - provide_data: same as DataIter, but specific to this batch
#   - provide_label: same as DataIter, but specific to this batch
#   - bucket_key: the key for the bucket that should be used for this batch

def default_read_content(path):
    with open(path) as ins:
        content = ins.read()
        content = content.replace('\n', ' <eos> ').replace('. ', ' <eos> ')
        return content

def default_build_vocab(path):
    content = default_read_content(path)
    content = content.split(' ')

    words = set([x for x in content if len(x) > 0])
    words = [x for x in words]
    words = sorted(words)
    the_vocab = {}
    idx = 1 # 0 is left for zero-padding
    the_vocab[' '] = 0 # put a dummy element here so that len(vocab) is correct
    for word in words:
        if len(word) == 0:
            continue
        if not word in the_vocab:
            the_vocab[word] = idx
            idx += 1
    return the_vocab

def default_text2id(sentence, the_vocab):
    words = sentence.split(' ')
    words = [the_vocab[w] for w in words if len(w) > 0]
    return words

def default_gen_buckets(sentences, batch_size, the_vocab):
    len_dict = {}
    max_len = -1
    for sentence in sentences:
        words = default_text2id(sentence, the_vocab)
        lw = len(words)
        if lw == 0:
            continue
        if lw > max_len:
            max_len = lw
        if lw in len_dict:
            len_dict[lw] += 1
        else:
            len_dict[lw] = 1
    print(len_dict)

    tl = 0
    buckets = []
    for l, n in len_dict.items(): # TODO: There are better heuristic ways to do this
        if n + tl >= batch_size:
            buckets.append(l)
            tl = 0
        else:
            tl += n
    if tl > 0:
        buckets.append(max_len)
    return buckets


class SimpleBatch(object):
    def __init__(self, data_names, data, label_names, label, bucket_key):
        self.data = data
        self.label = label
        self.data_names = data_names
        self.label_names = label_names
        self.bucket_key = bucket_key

        self.pad = 0
        self.index = None # TODO: what is index?

    @property
    def provide_data(self):
        return [(n, x.shape) for n, x in zip(self.data_names, self.data)]

    @property
    def provide_label(self):
        return [(n, x.shape) for n, x in zip(self.label_names, self.label)]

class DummyIter(mx.io.DataIter):
    "A dummy iterator that always return the same batch, used for speed testing"
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

class BucketSentenceIter(mx.io.DataIter):
    def __init__(self, path, vocab, buckets, batch_size,
                 init_states, data_name='data', label_name='label',
                 seperate_char=' <eos> ', text2id=None, read_content=None):
        super(BucketSentenceIter, self).__init__()

        if text2id is None:
            self.text2id = default_text2id
        else:
            self.text2id = text2id
        if read_content is None:
            self.read_content = default_read_content
        else:
            self.read_content = read_content
        content = self.read_content(path)
        sentences = content.split(seperate_char)

        if len(buckets) == 0:
            buckets = default_gen_buckets(sentences, batch_size, vocab)
        print(buckets)
        self.vocab_size = len(vocab)
        self.data_name = data_name
        self.label_name = label_name

        buckets.sort()
        self.buckets = buckets
        self.data = [[] for _ in buckets]

        # pre-allocate with the largest bucket for better memory sharing
        self.default_bucket_key = max(buckets)

        for sentence in sentences:
            sentence = self.text2id(sentence, vocab)
            if len(sentence) == 0:
                continue
            for i, bkt in enumerate(buckets):
                if bkt >= len(sentence):
                    self.data[i].append(sentence)
                    break
            # we just ignore the sentence it is longer than the maximum
            # bucket size here

        # convert data into ndarrays for better speed during training
        data = [np.zeros((len(x), buckets[i])) for i, x in enumerate(self.data)]
        for i_bucket in range(len(self.buckets)):
            for j in range(len(self.data[i_bucket])):
                sentence = self.data[i_bucket][j]
                data[i_bucket][j, :len(sentence)] = sentence
        self.data = data

        # Get the size of each bucket, so that we could sample
        # uniformly from the bucket
        bucket_sizes = [len(x) for x in self.data]

        print("Summary of dataset ==================")
        for bkt, size in zip(buckets, bucket_sizes):
            print("bucket of len %3d : %d samples" % (bkt, size))

        self.batch_size = batch_size
        self.make_data_iter_plan()

        self.init_states = init_states
        self.init_state_arrays = [mx.nd.zeros(x[1]) for x in init_states]

        self.provide_data = [('data', (batch_size, self.default_bucket_key))] + init_states
        self.provide_label = [('softmax_label', (self.batch_size, self.default_bucket_key))]

    def make_data_iter_plan(self):
        "make a random data iteration plan"
        # truncate each bucket into multiple of batch-size
        bucket_n_batches = []
        for i in range(len(self.data)):
            bucket_n_batches.append(len(self.data[i]) / self.batch_size)
            self.data[i] = self.data[i][:int(bucket_n_batches[i]*self.batch_size)]

        bucket_plan = np.hstack([np.zeros(n, int)+i for i, n in enumerate(bucket_n_batches)])
        np.random.shuffle(bucket_plan)

        bucket_idx_all = [np.random.permutation(len(x)) for x in self.data]

        self.bucket_plan = bucket_plan
        self.bucket_idx_all = bucket_idx_all
        self.bucket_curr_idx = [0 for x in self.data]

        self.data_buffer = []
        self.label_buffer = []
        for i_bucket in range(len(self.data)):
            data = np.zeros((self.batch_size, self.buckets[i_bucket]))
            label = np.zeros((self.batch_size, self.buckets[i_bucket]))
            self.data_buffer.append(data)
            self.label_buffer.append(label)

    def __iter__(self):
        init_state_names = [x[0] for x in self.init_states]

        for i_bucket in self.bucket_plan:
            data = self.data_buffer[i_bucket]
            label = self.label_buffer[i_bucket]

            i_idx = self.bucket_curr_idx[i_bucket]
            idx = self.bucket_idx_all[i_bucket][i_idx:i_idx+self.batch_size]
            self.bucket_curr_idx[i_bucket] += self.batch_size
            data[:] = self.data[i_bucket][idx]

            for k in range(len(data)):
                label[k] = sorted(data[k])
                #count = len(data[k]) / 2
                #for j in range(count):
                #    data[j+count] = data[j]

            #label[:, :-1] = data[:, 1:]
            #label[:, -1] = 0

            data_all = [mx.nd.array(data)] + self.init_state_arrays
            label_all = [mx.nd.array(label)]
            data_names = ['data'] + init_state_names
            label_names = ['softmax_label']

            data_batch = SimpleBatch(data_names, data_all, label_names, label_all,
                                     self.buckets[i_bucket])

            yield data_batch

    def reset(self):
        self.bucket_curr_idx = [0 for x in self.data]
