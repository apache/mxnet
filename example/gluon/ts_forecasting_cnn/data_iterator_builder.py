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


from mxnet import ndarray as nd
from mxnet import gluon
from utils import create_context


class DIterators(object):
    def __init__(self, options):
        self._options = options
        self.receptive_field = self._options.dilation_depth ** 2

    def build_iterator(self, data, for_train):
        ctx = create_context(self._options.num_gpu)

        T = data.shape[0]
        X3 = nd.zeros((T - self.receptive_field, data.shape[1], self.receptive_field), ctx=ctx)
        y = nd.zeros((T - self.receptive_field, data.shape[1]), ctx=ctx)

        for i in range(T - self.receptive_field):
            for j in range(data.shape[1]):
                X3[i, j, :] = data[i:i + self.receptive_field, j]
                y[i, j] = data[i + self.receptive_field, j]

        if self._options.model == 'cw':
            dataset = gluon.data.ArrayDataset(X3, y[:, self._options.trajectory])
        else:
            dataset = gluon.data.ArrayDataset(X3[:, self._options.trajectory, :], y[:, self._options.trajectory])

        if for_train:
            diter = gluon.data.DataLoader(dataset, self._options.batch_size,\
                                          shuffle=True, last_batch='discard')
        else:
            diter = gluon.data.DataLoader(dataset, self._options.batch_size_predict,\
                                          shuffle=False, last_batch='discard')
        return diter




