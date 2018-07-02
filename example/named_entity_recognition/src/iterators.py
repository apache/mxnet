# !/usr/bin/env python

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

# -*- coding: utf-8 -*-

import bisect
import random
import numpy as np
from mxnet.io import DataIter, DataBatch, DataDesc
from mxnet import ndarray
from sklearn.utils import shuffle

class BucketNerIter(DataIter):
    """
    This iterator can handle variable length feature/label arrays for MXNet RNN classifiers.
    This iterator can ingest 2d list of sentences, 2d list of entities and 3d list of characters.
    """

    def __init__(self, sentences, characters, label, max_token_chars, batch_size, buckets=None, data_pad=-1, label_pad = -1, data_names=['sentences', 'characters'],
                 label_name='seq_label', dtype='float32'):

        super(BucketNerIter, self).__init__()

        # Create a bucket for every seq length where there are more examples than the batch size
        if not buckets:
            seq_counts = np.bincount([len(s) for s in sentences])
            buckets = [i for i, j in enumerate(seq_counts) if j >= batch_size]
        buckets.sort()
        print("\nBuckets  created: ", buckets)
        assert(len(buckets) > 0), "Not enough utterances to create any buckets."

        ###########
        # Sentences
        ###########
        nslice = 0
        # Create empty nested lists for storing data that falls into each bucket
        self.sentences = [[] for _ in buckets]
        for i, sent in enumerate(sentences):
            # Find the index of the smallest bucket that is larger than the sentence length
            buck_idx = bisect.bisect_left(buckets, len(sent))

            if buck_idx == len(buckets): # If the sentence is larger than the largest bucket
                buck_idx = buck_idx - 1
                nslice += 1
                sent = sent[:buckets[buck_idx]] #Slice sentence to largest bucket size

            buff = np.full((buckets[buck_idx]), data_pad, dtype=dtype) # Create an array filled with 'data_pad'
            buff[:len(sent)] = sent # Fill with actual values
            self.sentences[buck_idx].append(buff) # Append array to index = bucket index
        self.sentences = [np.asarray(i, dtype=dtype) for i in self.sentences] # Convert to list of array
        print("Warning, {0} sentences sliced to largest bucket size.".format(nslice)) if nslice > 0 else None

        ############
        # Characters
        ############
        # Create empty nested lists for storing data that falls into each bucket
        self.characters = [[] for _ in buckets]
        for i, charsent in enumerate(characters):
            # Find the index of the smallest bucket that is larger than the sentence length
            buck_idx = bisect.bisect_left(buckets, len(charsent))

            if buck_idx == len(buckets): # If the sentence is larger than the largest bucket
                buck_idx = buck_idx - 1
                charsent = charsent[:buckets[buck_idx]] #Slice sentence to largest bucket size

            charsent = [word[:max_token_chars]for word in charsent] # Slice to max length
            charsent = [word + [data_pad]*(max_token_chars-len(word)) for word in charsent]# Pad to max length
            charsent = np.array(charsent)
            buff = np.full((buckets[buck_idx], max_token_chars), data_pad, dtype=dtype)
            buff[:charsent.shape[0], :] = charsent # Fill with actual values
            self.characters[buck_idx].append(buff) # Append array to index = bucket index
        self.characters = [np.asarray(i, dtype=dtype) for i in self.characters] # Convert to list of array

        ##########
        # Entities
        ##########
        # Create empty nested lists for storing data that falls into each bucket
        self.label = [[] for _ in buckets]
        self.indices = [[] for _ in buckets]
        for i, entities in enumerate(label):
            # Find the index of the smallest bucket that is larger than the sentence length
            buck_idx = bisect.bisect_left(buckets, len(entities))

            if buck_idx == len(buckets):  # If the sentence is larger than the largest bucket
                buck_idx = buck_idx - 1
                entities = entities[:buckets[buck_idx]]  # Slice sentence to largest bucket size

            buff = np.full((buckets[buck_idx]), label_pad, dtype=dtype)  # Create an array filled with 'data_pad'
            buff[:len(entities)] = entities  # Fill with actual values
            self.label[buck_idx].append(buff)  # Append array to index = bucket index
            self.indices[buck_idx].append(i)
        self.label = [np.asarray(i, dtype=dtype) for i in self.label]  # Convert to list of array
        self.indices = [np.asarray(i, dtype=dtype) for i in self.indices]  # Convert to list of array

        self.data_names = data_names
        self.label_name = label_name
        self.batch_size = batch_size
        self.max_token_chars = max_token_chars
        self.buckets = buckets
        self.dtype = dtype
        self.data_pad = data_pad
        self.label_pad = label_pad
        self.default_bucket_key = max(buckets)
        self.layout = 'NT'

        self.provide_data = [DataDesc(name=self.data_names[0], shape=(self.batch_size, self.default_bucket_key), layout=self.layout),
                             DataDesc(name=self.data_names[1], shape=(self.batch_size, self.default_bucket_key, self.max_token_chars), layout=self.layout)]
        self.provide_label=[DataDesc(name=self.label_name, shape=(self.batch_size, self.default_bucket_key), layout=self.layout)]

        #create empty list to store batch index values
        self.idx = []
        #for each bucketarray
        for i, buck in enumerate(self.sentences):
            #extend the list eg output with batch size 5 and 20 training examples in bucket. [(0,0), (0,5), (0,10), (0,15), (1,0), (1,5), (1,10), (1,15)]
            self.idx.extend([(i, j) for j in range(0, len(buck) - batch_size + 1, batch_size)])
        self.curr_idx = 0
        self.reset()

    def reset(self):
        """Resets the iterator to the beginning of the data."""
        self.curr_idx = 0
        #shuffle data in each bucket
        random.shuffle(self.idx)
        for i, buck in enumerate(self.sentences):
            self.indices[i], self.sentences[i], self.characters[i], self.label[i] = shuffle(self.indices[i],
                                                                                            self.sentences[i],
                                                                                            self.characters[i],
                                                                                            self.label[i])

        self.ndindex = []
        self.ndsent = []
        self.ndchar = []
        self.ndlabel = []

        #for each bucket of data
        for i, buck in enumerate(self.sentences):
            #append the lists with an array
            self.ndindex.append(ndarray.array(self.indices[i], dtype=self.dtype))
            self.ndsent.append(ndarray.array(self.sentences[i], dtype=self.dtype))
            self.ndchar.append(ndarray.array(self.characters[i], dtype=self.dtype))
            self.ndlabel.append(ndarray.array(self.label[i], dtype=self.dtype))

    def next(self):
        """Returns the next batch of data."""
        if self.curr_idx == len(self.idx):
            raise StopIteration
        #i = batches index, j = starting record
        i, j = self.idx[self.curr_idx] 
        self.curr_idx += 1

        indices = self.ndindex[i][j:j + self.batch_size]
        sentences = self.ndsent[i][j:j + self.batch_size]
        characters = self.ndchar[i][j:j + self.batch_size]
        label = self.ndlabel[i][j:j + self.batch_size]

        return DataBatch([sentences, characters], [label], pad=0, index = indices, bucket_key=self.buckets[i],
                         provide_data=[DataDesc(name=self.data_names[0], shape=sentences.shape, layout=self.layout),
                                       DataDesc(name=self.data_names[1], shape=characters.shape, layout=self.layout)],
                         provide_label=[DataDesc(name=self.label_name, shape=label.shape, layout=self.layout)])