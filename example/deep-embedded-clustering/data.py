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

# pylint: disable=missing-docstring
from __future__ import print_function

import numpy as np
import mxnet as mx


def get_mnist():
    """ Gets MNIST dataset """

    np.random.seed(1234) # set seed for deterministic ordering
    mnist_data = mx.test_utils.get_mnist()
    X = np.concatenate([mnist_data['train_data'], mnist_data['test_data']])
    Y = np.concatenate([mnist_data['train_label'], mnist_data['test_label']])
    p = np.random.permutation(X.shape[0])
    X = X[p].reshape((X.shape[0], -1)).astype(np.float32)*5
    Y = Y[p]
    return X, Y
