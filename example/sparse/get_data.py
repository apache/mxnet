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
