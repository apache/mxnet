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
# pylint: disable= arguments-differ, wildcard-import
"Bounding box transforms."
import random

from ....block import Block, HybridBlock
from ....nn import Sequential, HybridSequential
from .....util import is_np_array
from ..... import nd, npx

__all__ = ['RandomImageBboxFlipLeftRight']


class RandomImageBboxFlipLeftRight(Block):
    def __init__(self, p=0.5):
        super(RandomImageBboxFlipLeftRight, self).__init__()
        self.p = p

    def forward(self, img, bbox):
        if self.p <= 0:
            return img, bbox
        elif self.p >= 1:
            img = self._flip_image(img)
            bbox = self._flip_bbox(img, bbox)
            return img, bbox
        else:
            if self.p < random.random():
                return img, bbox
            else:
                img = self._flip_image(img)
                bbox = self._flip_bbox(img, bbox)
                return img, bbox

    def _flip_image(self, img):
        if is_np_array():
            return npx.image.flip_left_right(img)
        else:
            return nd.image.flip_left_right(img)

    def _flip_bbox(self, img, bbox):
        width = img.shape[-3]
        xmax = width - bbox[:, 0]
        xmin = width - bbox[:, 2]
        bbox[:, 0] = xmin
        bbox[:, 2] = xmax
        return bbox
            

