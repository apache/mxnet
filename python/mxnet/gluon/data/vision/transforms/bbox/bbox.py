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

from .....block import Block, HybridBlock
from .....nn import Sequential, HybridSequential
from ......util import is_np_array
from ...... import nd, npx, np
from .utils import bbox_crop, bbox_iou, bbox_random_crop_with_constraints

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
            

class ImageBboxCrop(Block):
    def __init__(self, crop, allow_outside_center=False):
        super(ImageBboxCrop, self).__init__()
        assert len(crop) == 4, "expect crop to be (x_min, y_min, x_max, y_max)"
        self.xmin = crop[0]
        self.ymin = crop[1]
        self.width = crop[2]
        self.height = crop[3]
        assert self.xmin >= 0
        assert self.ymin >= 0
        assert self.width > 0
        assert self.height > 0
        self.xmax = self.width + self.xmin
        self.ymax = self.height + self.ymin
        self._allow_outside_center = allow_outside_center

    def forward(self, img, bbox):
        if self.xmax >= img.shape[-2] or self.ymax < img.shape[-3]:
            return img, bbox
        if is_np_array():
            new_img = npx.image.crop(img, self.xmin, self.ymin, self.width, self.height)
            new_bbox = bbox_crop(bbox, (self.xmin, self.ymin, self.width, self.height), self._allow_outside_center)
        else:
            new_img = nd.image.crop(img, self.xmin, self.ymin, self.width, self.height)
            new_bbox = bbox_crop(bbox, (self.xmin, self.ymin, self.width, self.height), self._allow_outside_center)


class RandomImageBboxCropWithConstraints(Block):
    def __init__(self, p=0.5, min_scale=0.3, max_scale=1,
                 max_aspect_ratio=2, constraints=None,
                 max_trial=50):
        super(RandomImageBboxCropWithConstraints, self).__init__()
        self.p = p
        self._args = {
            "min_scale": min_scale,
            "max_scale": max_scale,
            "max_aspect_ratio": max_aspect_ratio,
            "constraints": constraints,
            "max_trial": max_trial
        }

    def forward(self, img, bbox):
        if random.random() < self.p:
            return img, bbox
        im_size = (img.shape[-2], img.shape[-3])
        new_bbox, crop = bbox_random_crop_with_constraints(bbox.asnumpy(), im_size, **self._args)
        if is_np_array():
            new_img = npx.image.crop(img, *crop)
            new_bbox = np.array(new_bbox)
        else:
            new_img = nd.image.crop(img, *crop)
            new_bbox = nd.array(new_bbox)
        return new_img, new_bbox