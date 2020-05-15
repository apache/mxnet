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

from .......base import numeric_types
from ......block import Block
from .......util import is_np_array
from ....... import ndarray as nd, numpy_extension as npx, numpy as np
from .utils import _check_bbox_shape, bbox_crop, bbox_translate
from .utils import bbox_resize, bbox_random_crop_with_constraints

__all__ = ['ImageBboxRandomFlipLeftRight', 'ImageBboxCrop',
           'ImageBboxRandomCropWithConstraints', 'ImageBboxResize']


class ImageBboxRandomFlipLeftRight(Block):
    """Randomly flip the input image and bbox left to right with a probability
    of p(0.5 by default).

    Parameters
    ----------
    p : float
        The probability to preceed with random cropping logic.

    Inputs:
        - **data**: input tensor with (Hi x Wi x C) shape.
        - **bbox**: input tensor with shape (N, 4+) where N is the number of bounding boxes.
            The second axis represents attributes of the bounding box.
            Specifically, these are :math:`(x_{min}, y_{min}, x_{max}, y_{max})`,
            we allow additional attributes other than coordinates, which stay intact
            during bounding box transformations.

    Outputs:
        - **out**: output tensor with same shape as `data`.
        - **bbox**: input tensor with same shape as `bbox`.
    """
    def __init__(self, p=0.5):
        super(ImageBboxRandomFlipLeftRight, self).__init__()
        self.p = p

    def forward(self, img, bbox):
        _check_bbox_shape(bbox)
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
        width = img.shape[-2]
        xmax = width - bbox[:, 0]
        xmin = width - bbox[:, 2]
        bbox[:, 0] = xmin
        bbox[:, 2] = xmax
        return bbox


class ImageBboxCrop(Block):
    """Crops the image `src` and `bbox` to the given `crop`.

    Parameters
    ----------
    crop_box : tuple
        Tuple of length 4. :math:`(x_{min}, y_{min}, width, height)`
    allow_outside_center : bool
        If `False`, remove bounding boxes which have centers outside cropping area.


    Inputs:
        - **data**: input tensor with (Hi x Wi x C) shape.
        - **bbox**: input tensor with shape (N, 4+) where N is the number of bounding boxes.
            The second axis represents attributes of the bounding box.
            Specifically, these are :math:`(x_{min}, y_{min}, x_{max}, y_{max})`,
            we allow additional attributes other than coordinates, which stay intact
            during bounding box transformations.

    Outputs:
        - **out**: output tensor with (H x W x C) shape.
        - **bbox**: output tensor with shape (M, 4+) where M <= N is the number of valid bounding
            boxes after cropping. :math:`(x_{min}, y_{min}, x_{max}, y_{max})`

    """
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
        if self.xmax >= img.shape[-2] or self.ymax >= img.shape[-3]:
            return img, bbox
        if is_np_array():
            new_img = npx.image.crop(img, self.xmin, self.ymin, self.width, self.height)
            new_bbox = np.array(bbox_crop(bbox.asnumpy(),
                                          (self.xmin, self.ymin, self.width, self.height),
                                          self._allow_outside_center))
        else:
            new_img = nd.image.crop(img, self.xmin, self.ymin, self.width, self.height)
            new_bbox = nd.array(bbox_crop(bbox.asnumpy(),
                                          (self.xmin, self.ymin, self.width, self.height),
                                          self._allow_outside_center))
        return new_img, new_bbox


class ImageBboxRandomCropWithConstraints(Block):
    """Crop an image randomly with bounding box constraints.

    Please check `mx.gluon.contrib.data.transforms.bbox.utils.bbox_random_crop_with_constraints`
    for implementation details.

    Parameters
    ----------
    p : float
        The probability to preceed with random cropping logic.
    min_scale : float
        The minimum ratio between a cropped region and the original image.
        The default value is :obj:`0.3`.
    max_scale : float
        The maximum ratio between a cropped region and the original image.
        The default value is :obj:`1`.
    max_aspect_ratio : float
        The maximum aspect ratio of cropped region.
        The default value is :obj:`2`.
    constraints : iterable of tuples
        An iterable of constraints.
        Each constraint should be :obj:`(min_iou, max_iou)` format.
        If means no constraint if set :obj:`min_iou` or :obj:`max_iou` to :obj:`None`.
        If this argument defaults to :obj:`None`, :obj:`((0.1, None), (0.3, None),
        (0.5, None), (0.7, None), (0.9, None), (None, 1))` will be used.
    max_trial : int
        Maximum number of trials for each constraint before exit no matter what.

    Inputs:
        - **data**: input tensor with (Hi x Wi x C) shape.
        - **bbox**: input tensor with shape (N, 4+) where N is the number of bounding boxes.
            The second axis represents attributes of the bounding box.
            Specifically, these are :math:`(x_{min}, y_{min}, x_{max}, y_{max})`,
            we allow additional attributes other than coordinates, which stay intact
            during bounding box transformations.

    Outputs:
        - **out**: Cropped image with shape (H x W x C)
        - **bbox**: Cropped bounding boxes with shape :obj:`(M, 4+)` where M <= N.
            Tuple of length 4 as :math:`(x_{min}, y_{min}, x_{max}, y_{max})`.
    """
    def __init__(self, p=0.5, min_scale=0.3, max_scale=1,
                 max_aspect_ratio=2, constraints=None,
                 max_trial=50):
        super(ImageBboxRandomCropWithConstraints, self).__init__()
        self.p = p
        self._args = {
            "min_scale": min_scale,
            "max_scale": max_scale,
            "max_aspect_ratio": max_aspect_ratio,
            "constraints": constraints,
            "max_trial": max_trial
        }

    def forward(self, img, bbox):
        if random.random() > self.p:
            return img, bbox
        im_size = (img.shape[-2], img.shape[-3])
        new_bbox, crop = bbox_random_crop_with_constraints(bbox.asnumpy(), im_size, **self._args)
        if crop == (0, 0, im_size[0], im_size[1]):
            return img, bbox
        if is_np_array():
            new_img = npx.image.crop(img, x=crop[0], y=crop[1], width=crop[2], height=crop[3])
            new_bbox = np.array(new_bbox)
        else:
            new_img = nd.image.crop(img, x=crop[0], y=crop[1], width=crop[2], height=crop[3])
            new_bbox = nd.array(new_bbox)
        return new_img, new_bbox


class ImageBboxRandomExpand(Block):
    """Randomly expand image to a larger region with padded pixels.
    Apply tranlation to bounding boxes accordingly.

    Parameters
    ----------
    p : float
        The probability to preceed with random cropping logic.
    max_ratio : float
        The minimum expansion ratio. If `max_ratio` is 2, the range of
        output image size is 1x ~ 2x of the original input size.
    fill : float or tuple of float
        The value(s) for the pixels in expanded regions. Can be scalar or tuple,
        note the if tuple is provided, its size must match the image channels, typically 3.
    keep_ratio : bool
        If `True`, the output must have the same aspect ratio as input, otherwise the output
        can have arbitrary aspect ratio.

    Inputs:
        - **data**: input tensor with (Hi x Wi x C) shape.
        - **bbox**: input tensor with shape (N, 4+) where N is the number of bounding boxes.
            The second axis represents attributes of the bounding box.
            Specifically, these are :math:`(x_{min}, y_{min}, x_{max}, y_{max})`,
            we allow additional attributes other than coordinates, which stay intact
            during bounding box transformations.

    Outputs:
        - **out**: Cropped image with shape (H x W x C)
        - **bbox**: Cropped bounding boxes with shape :obj:`(N, 4+)`.
            Tuple of length 4 as :math:`(x_{min}, y_{min}, x_{max}, y_{max})`.

    """
    def __init__(self, p=0.5, max_ratio=4, fill=0, keep_ratio=True):
        super(ImageBboxRandomExpand, self).__init__()
        self.p = p
        self._max_ratio = max_ratio
        self._fill = fill
        self._keep_ratio = keep_ratio

    def forward(self, img, bbox):
        if self._max_ratio <= 1 or random.random() > self.p:
            return img, bbox
        if len(img.shape) != 3:
            raise NotImplementedError('ImageBboxRandomExpand only support images in HWC format')

        h, w, c = img.shape
        ratio_x = random.uniform(1, self._max_ratio)
        if self._keep_ratio:
            ratio_y = ratio_x
        else:
            ratio_y = random.uniform(1, self._max_ratio)

        oh, ow = int(h * ratio_y), int(w * ratio_x)
        off_y = random.randint(0, oh - h)
        off_x = random.randint(0, ow - w)

        # make canvas
        if is_np_array():
            F = np
        else:
            F = nd
        if isinstance(self._fill, numeric_types):
            dst = F.full(shape=(oh, ow, c), val=self._fill, dtype=img.dtype)
        else:
            fill = F.array(self._fill, dtype=img.dtype, ctx=img.ctx)
            if not c == fill.size:
                raise ValueError("Channel and fill size mismatch, {} vs {}".format(c, fill.size))
            dst = F.tile(fill.reshape((1, c)), reps=(oh * ow, 1)).reshape((oh, ow, c))

        dst[off_y:off_y+h, off_x:off_x+w, :] = img

        # translate bbox
        new_bbox = bbox_translate(bbox.asnumpy(), off_x, off_y)
        if is_np_array():
            new_bbox = np.array(new_bbox)
        else:
            new_bbox = nd.array(new_bbox)

        return dst, new_bbox


class ImageBboxResize(Block):
    """Apply resize to image and bounding boxes.

    Parameters
    ----------
    width : int
        The target output width.
    height : int
        The target output height.

    Inputs:
        - **data**: input tensor with (Hi x Wi x C) shape.
        - **bbox**: input tensor with shape (N, 4+) where N is the number of bounding boxes.
            The second axis represents attributes of the bounding box.
            Specifically, these are :math:`(x_{min}, y_{min}, x_{max}, y_{max})`,
            we allow additional attributes other than coordinates, which stay intact
            during bounding box transformations.

    Outputs:
        - **out**: Cropped image with shape (H x W x C)
        - **bbox**: Cropped bounding boxes with shape :obj:`(M, 4+)` where M <= N.
            Tuple of length 4 as :math:`(x_{min}, y_{min}, x_{max}, y_{max})`.

    """
    def __init__(self, width, height, interp=1):
        super(ImageBboxResize, self).__init__()
        self._size = (width, height)
        self._interp = interp

    def forward(self, img, bbox):
        if len(img.shape) != 3:
            raise NotImplementedError('ImageBboxResize only support images in HWC format')

        if self._interp == -1:
            # random interpolation mode
            interp = random.randint(0, 5)
        else:
            interp = self._interp

        if is_np_array():
            new_img = npx.image.resize(img, self._size, False, interp)
            new_bbox = np.array(bbox_resize(bbox.asnumpy(),
                                            (img.shape[-2], img.shape[-3]), self._size))
        else:
            new_img = nd.image.resize(img, self._size, False, interp)
            new_bbox = nd.array(bbox_resize(bbox.asnumpy(),
                                            (img.shape[-2], img.shape[-3]), self._size))
        return new_img, new_bbox
