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

# coding: utf-8
"""Dynamic loss scaler for AMP."""
import logging

from ...ndarray import multi_all_finite
from ...ndarray import ndarray as nd
from ... import autograd as ag

class LossScaler(object):
    """Dynamic loss scaler for AMP.

    Properties
    ----------
    loss_scale : float
        The current loss scale
    """
    def __init__(self):
        self._loss_scale = 2.**16
        self._next_loss_scale = self._loss_scale
        self._max_loss_scale = 2.**24
        self._scale_seq_len = 2000
        self._unskipped = 0

    @property
    def loss_scale(self):
        return self._loss_scale

    def has_overflow(self, params):
        """Check gradients for overflow."""
        with ag.pause():
            chunk_size = 200
            valid_params = [p._grad[0] for p in params if p._grad is not None]
            gpu_output = nd.ones((1,), ctx=valid_params[0].context)
            nb_params = len(valid_params)
            for idx in range(0, nb_params, chunk_size):
                multi_all_finite(*valid_params[idx:idx+chunk_size],
                                 num_arrays=len(valid_params[idx:idx+chunk_size]),
                                 init_output=False, out=gpu_output)
        has_overflow = not bool(gpu_output.asnumpy())
        self._loss_scale = self._next_loss_scale
        if has_overflow:
            self._next_loss_scale = self._loss_scale / 2.
            self._unskipped = 0
            logging.info("AMP: decreasing loss scale to %f", self._next_loss_scale)
        else:
            self._unskipped += 1
        if self._unskipped == self._scale_seq_len:
            self._unskipped = 0
            self._next_loss_scale = min(self._max_loss_scale, self._loss_scale * 2.)
            logging.info("AMP: increasing loss scale to %f", self._next_loss_scale)
        return has_overflow
