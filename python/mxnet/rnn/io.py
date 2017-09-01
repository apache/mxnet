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

# coding: utf-8
# pylint: disable=too-many-arguments, too-many-locals
"""Definition of various recurrent neural network cells."""
from __future__ import print_function

import bisect
import random
import numpy as np

from ..io import DataIter, DataBatch, DataDesc
from .. import ndarray

def encode_sentences(sentences, vocab=None, invalid_label=-1, invalid_key='\n', start_label=0):
    """Encode sentences and (optionally) build a mapping
    from string tokens to integer indices. Unknown keys
    will be added to vocabulary.

    Parameters
    ----------
    sentences : list of list of str
        A list of sentences to encode. Each sentence
        should be a list of string tokens.
    vocab : None or dict of str -> int
        Optional input Vocabulary
    invalid_label : int, default -1
        Index for invalid token, like <end-of-sentence>
    invalid_key : str, default '\\n'
        Key for invalid token. Use '\\n' for end
        of sentence by default.
    start_label : int
        lowest index.

    Returns
    -------
    result : list of list of int
        encoded sentences
    vocab : dict of str -> int
        result vocabulary
    """
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
                assert new_vocab, "Unknown token %s"%word
                if idx == invalid_label:
                    idx += 1
                vocab[word] = idx
                idx += 1
            coded.append(vocab[word])
        res.append(coded)

    return res, vocab

class BucketSentenceIter(DataIter):
    """Simple bucketing iterator for language model.
    The label at each sequence step is the following token
    in the sequence.

    Parameters
    ----------
    sentences : list of list of int
        Encoded sentences.
    batch_size : int
        Batch size of the data.
    invalid_label : int, optional
        Key for invalid label, e.g. <end-of-sentence>. The default is -1.
    dtype : str, optional
        Data type of the encoding. The default data type is 'float32'.
    buckets : list of int, optional
        Size of the data buckets. Automatically generated if None.
    data_name : str, optional
        Name of the data. The default name is 'data'.
    label_name : str, optional
        Name of the label. The default name is 'softmax_label'.
    layout : str, optional
        Format of data and label. 'NT' means (batch_size, length)
        and 'TN' means (length, batch_size).
    """
    def __init__(self, sentences, batch_size, buckets=None, invalid_label=-1,
                 data_name='data', label_name='softmax_label', dtype='float32',
                 layout='NT'):
        super(BucketSentenceIter, self).__init__()
        if not buckets:
            buckets = [i for i, j in enumerate(np.bincount([len(s) for s in sentences]))
                       if j >= batch_size]
        buckets.sort()

        ndiscard = 0
        self.data = [[] for _ in buckets]
        for i, sent in enumerate(sentences):
            buck = bisect.bisect_left(buckets, len(sent))
            if buck == len(buckets):
                ndiscard += 1
                continue
            buff = np.full((buckets[buck],), invalid_label, dtype=dtype)
            buff[:len(sent)] = sent
            self.data[buck].append(buff)

        self.data = [np.asarray(i, dtype=dtype) for i in self.data]

        print("WARNING: discarded %d sentences longer than the largest bucket."%ndiscard)

        self.batch_size = batch_size
        self.buckets = buckets
        self.data_name = data_name
        self.label_name = label_name
        self.dtype = dtype
        self.invalid_label = invalid_label
        self.nddata = []
        self.ndlabel = []
        self.major_axis = layout.find('N')
        self.layout = layout
        self.default_bucket_key = max(buckets)

        if self.major_axis == 0:
            self.provide_data = [DataDesc(
                name=self.data_name, shape=(batch_size, self.default_bucket_key),
                layout=self.layout)]
            self.provide_label = [DataDesc(
                name=self.label_name, shape=(batch_size, self.default_bucket_key),
                layout=self.layout)]
        elif self.major_axis == 1:
            self.provide_data = [DataDesc(
                name=self.data_name, shape=(self.default_bucket_key, batch_size),
                layout=self.layout)]
            self.provide_label = [DataDesc(
                name=self.label_name, shape=(self.default_bucket_key, batch_size),
                layout=self.layout)]
        else:
            raise ValueError("Invalid layout %s: Must by NT (batch major) or TN (time major)")

        self.idx = []
        for i, buck in enumerate(self.data):
            self.idx.extend([(i, j) for j in range(0, len(buck) - batch_size + 1, batch_size)])
        self.curr_idx = 0

        self.reset()

    def reset(self):
        """Resets the iterator to the beginning of the data."""
        self.curr_idx = 0
        random.shuffle(self.idx)
        for buck in self.data:
            np.random.shuffle(buck)

        self.nddata = []
        self.ndlabel = []
        for buck in self.data:
            label = np.empty_like(buck)
            label[:, :-1] = buck[:, 1:]
            label[:, -1] = self.invalid_label
            self.nddata.append(ndarray.array(buck, dtype=self.dtype))
            self.ndlabel.append(ndarray.array(label, dtype=self.dtype))

    def next(self):
        """Returns the next batch of data."""
        if self.curr_idx == len(self.idx):
            raise StopIteration
        i, j = self.idx[self.curr_idx]
        self.curr_idx += 1

        if self.major_axis == 1:
            data = self.nddata[i][j:j+self.batch_size].T
            label = self.ndlabel[i][j:j+self.batch_size].T
        else:
            data = self.nddata[i][j:j+self.batch_size]
            label = self.ndlabel[i][j:j+self.batch_size]

        return DataBatch([data], [label], pad=0,
                         bucket_key=self.buckets[i],
                         provide_data=[DataDesc(
                             name=self.data_name, shape=data.shape,
                             layout=self.layout)],
                         provide_label=[DataDesc(
                             name=self.label_name, shape=label.shape,
                             layout=self.layout)])
