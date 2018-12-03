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

"""This module builds a model an MLP with a configurable output layer( number of units in the last layer).
Users can pass any number of units in the last layer. SInce this dataset has 10 labels,
the default value of num_labels = 10
"""
import mxnet as mx
from mxnet import gluon

# Defining a neural network with number of labels
def get_net(num_labels=10):
    net = gluon.nn.Sequential()
    with net.name_scope():
        net.add(gluon.nn.Dense(256, activation="relu")) # 1st layer (256 nodes)
        net.add(gluon.nn.Dense(256, activation="relu")) # 2nd hidden layer ( 256 nodes )
        net.add(gluon.nn.Dense(num_labels))
    net.collect_params().initialize(mx.init.Xavier())
    return net
