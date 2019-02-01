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
""" Iterator for Captcha images used for LSTM-based OCR model"""

from __future__ import print_function

import numpy as np
import mxnet as mx


class SimpleBatch(object):
    """Batch class for getting label data
    Operation:
        - call get_label() to start label data generation
    """
    def __init__(self, data_names, data, label_names=None, label=None):
        self._data = data
        self._label = label
        self._data_names = data_names
        self._label_names = label_names

        self.pad = 0
        self.index = None  # TODO: what is index?

    @property
    def data(self):
        return self._data

    @property
    def label(self):
        return self._label

    @property
    def data_names(self):
        return self._data_names

    @property
    def label_names(self):
        return self._label_names

    @property
    def provide_data(self):
        return [(n, x.shape) for n, x in zip(self._data_names, self._data)]

    @property
    def provide_label(self):
        return [(n, x.shape) for n, x in zip(self._label_names, self._label)]


def get_label(buf):
    ret = np.zeros(4)
    for i, element in enumerate(buf):
        ret[i] = 1 + int(element)
    if len(buf) == 3:
        ret[3] = 0
    return ret


class OCRIter(mx.io.DataIter):
    """Iterator class for generating captcha image data"""
    def __init__(self, count, batch_size, lstm_init_states, captcha, name):
        """Parameters
        ----------
        count: int
            Number of batches to produce for one epoch
        batch_size: int
        lstm_init_states: list of tuple(str, tuple)
            A list of tuples with [0] name and [1] shape of each LSTM init state
        captcha MPCaptcha
            Captcha image generator. Can be MPCaptcha or any other class providing .shape and .get() interface
        name: str
        """
        super(OCRIter, self).__init__()
        self.batch_size = batch_size
        self.count = count
        self.init_states = lstm_init_states
        self.init_state_arrays = [mx.nd.zeros(x[1]) for x in lstm_init_states]
        data_shape = captcha.shape
        self.provide_data = [('data', (batch_size, data_shape[0], data_shape[1]))] + lstm_init_states
        self.provide_label = [('label', (self.batch_size, 4))]
        self.mp_captcha = captcha
        self.name = name

    def __iter__(self):
        init_state_names = [x[0] for x in self.init_states]
        for k in range(self.count):
            data = []
            label = []
            for i in range(self.batch_size):
                img, num = self.mp_captcha.get()
                data.append(img)
                label.append(get_label(num))
            data_all = [mx.nd.array(data)] + self.init_state_arrays
            label_all = [mx.nd.array(label)]
            data_names = ['data'] + init_state_names
            label_names = ['label']

            data_batch = SimpleBatch(data_names, data_all, label_names, label_all)
            yield data_batch
