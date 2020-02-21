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
# pylint: disable=wildcard-import
"""Contrib vision trasforms."""
import random
from ....block import Block, HybridBlock
from ....nn import Sequential, HybridSequential
from ..... import image
from .....base import numeric_types
from .....util import is_np_array


class BBoxRandomFlipLeftRight(HybridBlock):
    """Randomly flip the input image left to right with a probability
    of 0.5.

    Inputs:
        - **data**: input tensor with (H x W x C) shape.

    Outputs:
        - **out**: output tensor with same shape as `data`.
    """
    def __init__(self, prob=0.5):
        super(BBoxRandomFlipLeftRight, self).__init__()
        self.prob = prob

    def hybrid_forward(self, F, x, y):
        if is_np_array():
            width = F.npx.shape_array(x).split(3)[1]
            cond = F.np.random.uniform(low=0, high=1, size=1) < self.prob
            x = F.np.where(cond, F.npx.image.flip_left_right(x), x)
        else:
            raise NotImplementedError('Not implemented for non-np mode')