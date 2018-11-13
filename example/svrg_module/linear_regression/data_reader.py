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

import bz2
import os
import shutil

import mxnet as mx
import numpy as np
from sklearn.datasets import load_svmlight_file


# Download data file
# YearPredictionMSD dataset: https://archive.ics.uci.edu/ml/datasets/yearpredictionmsd


def get_year_prediction_data(dirname=None):
    feature_dim = 90
    if dirname is None:
        dirname = os.path.join(os.path.dirname(__file__), 'data')
    filename = 'YearPredictionMSD'
    download_filename = os.path.join(dirname, "%s.bz2" % filename)
    extracted_filename = os.path.join(dirname, filename)
    if not os.path.isfile(download_filename):
        print("Downloading data...")
        mx.test_utils.download('https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/regression/%s.bz2' % filename, dirname=dirname)
    if not os.path.isfile(extracted_filename):
        print("Extracting data...")
        with bz2.BZ2File(download_filename) as fr, open(extracted_filename,"wb") as fw:
            shutil.copyfileobj(fr,fw)
    print("Reading data from disk...")
    train_features, train_labels = load_svmlight_file(extracted_filename, n_features=feature_dim, dtype=np.float32)
    train_features = train_features.todense()

    # normalize the data: subtract means and divide by standard deviations
    label_mean = train_labels.mean()
    label_std = np.sqrt(np.square(train_labels - label_mean).mean())
    feature_means = train_features.mean(axis=0)
    feature_stds = np.sqrt(np.square(train_features - feature_means).mean(axis=0))

    train_features = (train_features - feature_means) / feature_stds
    train_labels = (train_labels - label_mean) / label_std

    return feature_dim, train_features, train_labels

