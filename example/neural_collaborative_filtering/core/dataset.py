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
#
import os 
import mxnet as mx
import numpy as np
import pandas as pd
import scipy.sparse as sp

class NCFTestData(object):
    def __init__(self, path):
        '''
        Constructor
        path: converted data root
        testRatings: converted test ratings data
        testNegatives: negative samples for evaluation dataset
        '''
        self.testRatings = self.load_rating_file_as_list(os.path.join(path, 'test-ratings.csv'))
        self.testNegatives = self.load_negative_file(os.path.join(path ,'test-negative.csv'))
        assert len(self.testRatings) == len(self.testNegatives)

    def load_rating_file_as_list(self, filename):
        ratingList = []
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split("\t")
                user, item = int(arr[0]), int(arr[1])
                ratingList.append([user, item])
                line = f.readline()
        return ratingList
    
    def load_negative_file(self, filename):
        negativeList = []
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split("\t")
                negatives = []
                for x in arr:
                    negatives.append(int(x))
                negativeList.append(negatives)
                line = f.readline()
        return negativeList

class NCFTrainData(mx.gluon.data.Dataset):
    def __init__(self, train_fname, nb_neg):
        '''
        Constructor
        train_fname: converted data root
        nb_neg: number of negative samples per positive sample while training
        '''
        self._load_train_matrix(train_fname)
        self.nb_neg = nb_neg

    def _load_train_matrix(self, train_fname):
        def process_line(line):
            tmp = line.split('\t')
            return [int(tmp[0]), int(tmp[1]), float(tmp[2]) > 0]
        with open(train_fname, 'r') as file:
            data = list(map(process_line, file))
        self.nb_users = max(data, key=lambda x: x[0])[0] + 1
        self.nb_items = max(data, key=lambda x: x[1])[1] + 1

        self.data = list(filter(lambda x: x[2], data))
        self.mat = sp.dok_matrix(
                (self.nb_users, self.nb_items), dtype=np.float32)
        for user, item, _ in data:
            self.mat[user, item] = 1.

    def __len__(self):
        return (self.nb_neg + 1) * len(self.data)

    def __getitem__(self, idx):
        if idx % (self.nb_neg + 1) == 0:
            idx = idx // (self.nb_neg + 1)
            return self.data[idx][0], self.data[idx][1], np.ones(1, dtype=np.float32).item()  # noqa: E501
        else:
            idx = idx // (self.nb_neg + 1)
            u = self.data[idx][0]
            j = mx.random.randint(0, self.nb_items).asnumpy().item()
            while (u, j) in self.mat:
                j = mx.random.randint(0, self.nb_items).asnumpy().item()
            return u, j, np.zeros(1, dtype=np.float32).item()

