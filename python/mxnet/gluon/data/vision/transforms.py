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
# pylint: disable= arguments-differ
"Image transforms."

from ...block import Block, HybridBlock
from ...nn import Sequential, HybridSequential
from .... import image
from ....base import numeric_types


class Compose(Sequential):
    """Sequentially composes multiple transforms.

    Parameters
    ----------
    transforms : list of transform Blocks.
        The list of transforms to be composed.
    """
    def __init__(self, transforms):
        super(Compose, self).__init__()
        transforms.append(None)
        hybrid = []
        for i in transforms:
            if isinstance(i, HybridBlock):
                hybrid.append(i)
                continue
            elif len(hybrid) == 1:
                self.add(hybrid[0])
            elif len(hybrid) > 1:
                hblock = HybridSequential()
                for j in hybrid:
                    hblock.add(j)
                self.add(hblock)
            if i is not None:
                self.add(i)
        self.hybridize()


class Cast(HybridBlock):
    """Cast input to a specific data type

    Parameters
    ----------
    dtype : str, default 'float32'
        The target data type, in string or `numpy.dtype`.
    """
    def __init__(self, dtype='float32'):
        super(Cast, self).__init__()
        self._dtype = dtype

    def hybrid_forward(self, F, x):
        return F.cast(x, self._dtype)


class ToTensor(HybridBlock):
    """Converts an image NDArray to a tensor NDArray.

    Converts an image NDArray of shape (H x W x C) in the range
    [0, 255] to a float32 tensor NDArray of shape (C x H x W) in
    the range [0, 1).
    """
    def __init__(self):
        super(ToTensor, self).__init__()

    def hybrid_forward(self, F, x):
        return F.image.to_tensor(x)


class Normalize(HybridBlock):
    """Normalize an tensor of shape (C x H x W) with mean and
    standard deviation.

    Given mean `(m1, ..., mn)` and std `(s1, ..., sn)` for `n` channels,
    this transform normalizes each channel of the input tensor with::

        output[i] = (input[i] - mi) / si

    If mean or std is scalar, the same value will be applied to all channels.

    Parameters
    ----------
    mean : float or tuple of floats
        The mean values.
    std : float or tuple of floats
        The standard deviation values.
    """
    def __init__(self, mean, std):
        super(Normalize, self).__init__()
        self._mean = mean
        self._std = std

    def hybrid_forward(self, F, x):
        return F.image.normalize(x, self._mean, self._std)


class RandomResizedCrop(Block):
    """Crop the input image with random scale and aspect ratio.

    Makes a crop of the original image with random size (default: 0.08
    to 1.0 of the original image size) and random aspect ratio (default:
    3/4 to 4/3), then resize it to the specified size.

    Parameters
    ----------
    size : int or tuple of (W, H)
        Size of the final output.
    scale : tuple of two floats
        If scale is `(min_area, max_area)`, the cropped image's area will
        range from min_area to max_area of the original image's area
    ratio : tuple of two floats
        Range of aspect ratio of the cropped image before resizing.
    interpolation : int
        Interpolation method for resizing. By default uses bilinear
        interpolation. See OpenCV's resize function for available choices.
    """
    def __init__(self, size, scale=(0.08, 1.0), ratio=(3.0/4.0, 4.0/3.0),
                 interpolation=2):
        super(RandomResizedCrop, self).__init__()
        if isinstance(size, numeric_types):
            size = (size, size)
        self._args = (size, scale[0], ratio, interpolation)

    def forward(self, x):
        return image.random_size_crop(x, *self._args)[0]


class CenterCrop(Block):
    """Crops the image `src` to the given `size` by trimming on all four
    sides and preserving the center of the image. Upsamples if `src` is
    smaller than `size`.

    Parameters
    ----------
    size : int or tuple of (W, H)
        Size of output image.
    interpolation : int
        Interpolation method for resizing. By default uses bilinear
        interpolation. See OpenCV's resize function for available choices.
    """
    def __init__(self, size, interpolation=2):
        super(CenterCrop, self).__init__()
        if isinstance(size, numeric_types):
            size = (size, size)
        self._args = (size, interpolation)

    def forward(self, x):
        return image.center_crop(x, *self._args)[0]


class Resize(Block):
    """Resize an image to the given size.

    Parameters
    ----------
    size : int or tuple of (W, H)
        Size of output image.
    interpolation : int
        Interpolation method for resizing. By default uses bilinear
        interpolation. See OpenCV's resize function for available choices.
    """
    def __init__(self, size, interpolation=2):
        super(Resize, self).__init__()
        if isinstance(size, numeric_types):
            size = (size, size)
        self._args = tuple(size) + (interpolation,)

    def forward(self, x):
        return image.imresize(x, *self._args)


class RandomFlipLeftRight(HybridBlock):
    """Randomly flip the input image left to right with a probability
    of 0.5.
    """
    def __init__(self):
        super(RandomFlipLeftRight, self).__init__()

    def hybrid_forward(self, F, x):
        return F.image.random_flip_left_right(x)


class RandomFlipTopBottom(HybridBlock):
    """Randomly flip the input image top to bottom with a probability
    of 0.5.
    """
    def __init__(self):
        super(RandomFlipTopBottom, self).__init__()

    def hybrid_forward(self, F, x):
        return F.image.random_flip_top_bottom(x)


class RandomBrightness(HybridBlock):
    """Randomly jitters image brightness with a factor
    chosen from `[max(0, 1 - brightness), 1 + brightness]`.
    """
    def __init__(self, brightness):
        super(RandomBrightness, self).__init__()
        self._args = (max(0, 1-brightness), 1+brightness)

    def hybrid_forward(self, F, x):
        return F.image.random_brightness(x, *self._args)


class RandomContrast(HybridBlock):
    """Randomly jitters image contrast with a factor
    chosen from `[max(0, 1 - contrast), 1 + contrast]`.
    """
    def __init__(self, contrast):
        super(RandomContrast, self).__init__()
        self._args = (max(0, 1-contrast), 1+contrast)

    def hybrid_forward(self, F, x):
        return F.image.random_contrast(x, *self._args)


class RandomSaturation(HybridBlock):
    """Randomly jitters image saturation with a factor
    chosen from `[max(0, 1 - saturation), 1 + saturation]`.
    """
    def __init__(self, saturation):
        super(RandomSaturation, self).__init__()
        self._args = (max(0, 1-saturation), 1+saturation)

    def hybrid_forward(self, F, x):
        return F.image.random_saturation(x, *self._args)


class RandomHue(HybridBlock):
    """Randomly jitters image hue with a factor
    chosen from `[max(0, 1 - hue), 1 + hue]`.
    """
    def __init__(self, hue):
        super(RandomHue, self).__init__()
        self._args = (max(0, 1-hue), 1+hue)

    def hybrid_forward(self, F, x):
        return F.image.random_hue(x, *self._args)


class RandomColorJitter(HybridBlock):
    """Randomly jitters the brightness, contrast, saturation, and hue
    of an image.

    Parameters
    ----------
    brightness : float
        How much to jitter brightness. brightness factor is randomly
        chosen from `[max(0, 1 - brightness), 1 + brightness]`.
    contrast : float
        How much to jitter contrast. contrast factor is randomly
        chosen from `[max(0, 1 - contrast), 1 + contrast]`.
    saturation : float
        How much to jitter saturation. saturation factor is randomly
        chosen from `[max(0, 1 - saturation), 1 + saturation]`.
    hue : float
        How much to jitter hue. hue factor is randomly
        chosen from `[max(0, 1 - hue), 1 + hue]`.
    """
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        super(RandomColorJitter, self).__init__()
        self._args = (brightness, contrast, saturation, hue)

    def hybrid_forward(self, F, x):
        return F.image.random_color_jitter(x, *self._args)


class RandomLighting(HybridBlock):
    """Add AlexNet-style PCA-based noise to an image.

    Parameters
    ----------
    alpha : float
        Intensity of the image.
    """
    def __init__(self, alpha):
        super(RandomLighting, self).__init__()
        self._alpha = alpha

    def hybrid_forward(self, F, x):
        return F.image.random_lighting(x, self._alpha)
