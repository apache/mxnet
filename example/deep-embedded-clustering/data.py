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

import os
import numpy as np
from sklearn.datasets import fetch_mldata


def get_mnist():
    """ Gets MNIST dataset """

    np.random.seed(1234) # set seed for deterministic ordering
    data_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
    data_path = os.path.join(data_path, '../../data')
    mnist = fetch_mldata('MNIST original', data_home=data_path)
    p = np.random.permutation(mnist.data.shape[0])
    X = mnist.data[p].astype(np.float32)*0.02
    Y = mnist.target[p]
    return X, Y




