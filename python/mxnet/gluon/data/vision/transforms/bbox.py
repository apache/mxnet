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
from ..... import nd, npx, np

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
            new_bbox = _crop_bbox(bbox, (self.xmin, self.ymin, self.width, self.height), self._allow_outside_center)
        else:
            new_img = nd.image.crop(img, self.xmin, self.ymin, self.width, self.height)
            new_bbox = _crop_bbox(bbox, (self.xmin, self.ymin, self.width, self.height), self._allow_outside_center)



class RandomImageBboxCropWithConstraints(Block):
    pass


def _crop_bbox(bbox, crop_box=None, allow_outside_center=True):
    """Crop bounding boxes according to slice area.
    This method is mainly used with image cropping to ensure bonding boxes fit
    within the cropped image.
    Parameters
    ----------
    bbox : numpy.ndarray
        Numpy.ndarray with shape (N, 4+) where N is the number of bounding boxes.
        The second axis represents attributes of the bounding box.
        Specifically, these are :math:`(x_{min}, y_{min}, x_{max}, y_{max})`,
        we allow additional attributes other than coordinates, which stay intact
        during bounding box transformations.
    crop_box : tuple
        Tuple of length 4. :math:`(x_{min}, y_{min}, width, height)`
    allow_outside_center : bool
        If `False`, remove bounding boxes which have centers outside cropping area.
    Returns
    -------
    numpy.ndarray
        Cropped bounding boxes with shape (M, 4+) where M <= N.
    """
    bbox = bbox.copy()
    if crop_box is None:
        return bbox
    if not len(crop_box) == 4:
        raise ValueError(
            "Invalid crop_box parameter, requires length 4, given {}".format(str(crop_box)))
    if sum([int(c is None) for c in crop_box]) == 4:
        return bbox

    l, t, w, h = crop_box

    left = l if l else 0
    top = t if t else 0
    right = left + (w if w else np.inf)
    bottom = top + (h if h else np.inf)
    crop_bbox = np.array((left, top, right, bottom))

    if allow_outside_center:
        mask = np.ones(bbox.shape[0], dtype=bool)
    else:
        centers = (bbox[:, :2] + bbox[:, 2:4]) / 2
        mask = ((crop_bbox[:2] <= centers) * (centers < crop_bbox[2:])).all(axis=1)

    # transform borders
    bbox[:, :2] = np.maximum(bbox[:, :2], crop_bbox[:2])
    bbox[:, 2:4] = np.minimum(bbox[:, 2:4], crop_bbox[2:4])
    bbox[:, :2] -= crop_bbox[:2]
    bbox[:, 2:4] -= crop_bbox[:2]

    mask = (mask * (bbox[:, :2] < bbox[:, 2:4]).all(axis=1))
    bbox = bbox[mask]
    return bbox