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


import numpy as np
from sklearn.datasets import load_svmlight_file

# Download data file
# from subprocess import call
# YearPredictionMSD dataset: https://archive.ics.uci.edu/ml/datasets/yearpredictionmsd
# call(['wget', 'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/regression/YearPredictionMSD.bz2'])
# call(['bzip2', '-d', 'YearPredictionMSD.bz2'])


def read_year_prediction_data(fileName):
    feature_dim = 90
    print("Reading data from disk...")
    train_features, train_labels = load_svmlight_file(fileName, n_features=feature_dim, dtype=np.float32)
    train_features = train_features.todense()

    # normalize the data: subtract means and divide by standard deviations
    label_mean = train_labels.mean()
    label_std = np.sqrt(np.square(train_labels - label_mean).mean())
    feature_means = train_features.mean(axis=0)
    feature_stds = np.sqrt(np.square(train_features - feature_means).mean(axis=0))

    train_features = (train_features - feature_means) / feature_stds
    train_labels = (train_labels - label_mean) / label_std

    return feature_dim, train_features, train_labels

