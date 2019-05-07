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

import os

import numpy as np
from net_builder import LorenzBuilder
from utils import rmse, plot_predictions, create_context

class Predict(object):
    """

    The modeling engine to predict with existing neural network.
    """
    def __init__(self, options):
        self._options = options

    def predict(self, predict_iter):
        ctx = create_context(self._options.num_gpu)
        net = LorenzBuilder(self._options, ctx=ctx, for_train=False).build()

        labels = []
        preds = []

        for x, y in predict_iter:
            x = x.as_in_context(ctx).reshape((x.shape[0], self._options.in_channels, -1))
            y = y.as_in_context(ctx)
            y_hat = net(x)
            preds.extend(y_hat.asnumpy().tolist()[0])
            labels.extend(y.asnumpy().tolist())

        np.savetxt(os.path.join(self._options.assets_dir, 'preds.txt'), preds)
        np.savetxt(os.path.join(self._options.assets_dir, 'labels.txt'), labels)




