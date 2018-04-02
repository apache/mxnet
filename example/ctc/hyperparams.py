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
""" Hyperparameters for LSTM OCR Example """

from __future__ import print_function


class Hyperparams(object):
    """
    Hyperparameters for LSTM network
    """
    def __init__(self):
        # Training hyper parameters
        self._train_epoch_size = 30000
        self._eval_epoch_size = 3000
        self._batch_size = 128
        self._num_epoch = 100
        self._learning_rate = 0.001
        self._momentum = 0.9
        self._num_label = 4
        # Network hyper parameters
        self._seq_length = 80
        self._num_hidden = 100
        self._num_lstm_layer = 2

    @property
    def train_epoch_size(self):
        return self._train_epoch_size

    @property
    def eval_epoch_size(self):
        return self._eval_epoch_size

    @property
    def batch_size(self):
        return self._batch_size
    
    @property
    def num_epoch(self):
        return self._num_epoch

    @property
    def learning_rate(self):
        return self._learning_rate

    @property
    def momentum(self):
        return self._momentum

    @property
    def num_label(self):
        return self._num_label

    @property
    def seq_length(self):
        return self._seq_length

    @property
    def num_hidden(self):
        return self._num_hidden

    @property
    def num_lstm_layer(self):
        return self._num_lstm_layer
