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
from ...utils import _adapt_np_array


class Compose(Sequential):
    """Sequentially composes multiple transforms.

    Parameters
    ----------
    transforms : list of transform Blocks.
        The list of transforms to be composed.


    Inputs:
        - **data**: input tensor with shape of the first transform Block requires.

    Outputs:
        - **out**: output tensor with shape of the last transform Block produces.

    Examples
    --------
    >>> transformer = transforms.Compose([transforms.Resize(300),
    ...                                   transforms.CenterCrop(256),
    ...                                   transforms.ToTensor()])
    >>> image = mx.nd.random.uniform(0, 255, (224, 224, 3)).astype(dtype=np.uint8)
    >>> transformer(image)
    <NDArray 3x256x256 @cpu(0)>
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
                hybrid = []
            elif len(hybrid) > 1:
                hblock = HybridSequential()
                for j in hybrid:
                    hblock.add(j)
                hblock.hybridize()
                self.add(hblock)
                hybrid = []

            if i is not None:
                self.add(i)


class Cast(HybridBlock):
    """Cast input to a specific data type

    Parameters
    ----------
    dtype : str, default 'float32'
        The target data type, in string or `numpy.dtype`.


    Inputs:
        - **data**: input tensor with arbitrary shape and dtype.

    Outputs:
        - **out**: output tensor with the same shape as `data` and data type as dtype.
    """
    def __init__(self, dtype='float32'):
        super(Cast, self).__init__()
        self._dtype = dtype

    def hybrid_forward(self, F, x):
        return F.cast(x, self._dtype)


class ToTensor(HybridBlock):
    """Converts an image NDArray or batch of image NDArray to a tensor NDArray.

    Converts an image NDArray of shape (H x W x C) in the range
    [0, 255] to a float32 tensor NDArray of shape (C x H x W) in
    the range [0, 1].

    If batch input, converts a batch image NDArray of shape (N x H x W x C) in the
    range [0, 255] to a float32 tensor NDArray of shape (N x C x H x W).

    Inputs:
        - **data**: input tensor with (H x W x C) or (N x H x W x C) shape and uint8 type.

    Outputs:
        - **out**: output tensor with (C x H x W) or (N x H x W x C) shape and float32 type.

    Examples
    --------
    >>> transformer = vision.transforms.ToTensor()
    >>> image = mx.nd.random.uniform(0, 255, (4, 2, 3)).astype(dtype=np.uint8)
    >>> transformer(image)
    [[[ 0.85490197  0.72156864]
      [ 0.09019608  0.74117649]
      [ 0.61960787  0.92941177]
      [ 0.96470588  0.1882353 ]]
     [[ 0.6156863   0.73725492]
      [ 0.46666667  0.98039216]
      [ 0.44705883  0.45490196]
      [ 0.01960784  0.8509804 ]]
     [[ 0.39607844  0.03137255]
      [ 0.72156864  0.52941179]
      [ 0.16470589  0.7647059 ]
      [ 0.05490196  0.70588237]]]
    <NDArray 3x4x2 @cpu(0)>
    """
    def __init__(self):
        super(ToTensor, self).__init__()

    @_adapt_np_array
    def hybrid_forward(self, F, x):
        return F.image.to_tensor(x)


class Normalize(HybridBlock):
    """Normalize an tensor of shape (C x H x W) or (N x C x H x W) with mean and
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


    Inputs:
        - **data**: input tensor with (C x H x W) or (N x C x H x W) shape.

    Outputs:
        - **out**: output tensor with the shape as `data`.

    Examples
    --------
    >>> transformer = transforms.Normalize(mean=(0, 1, 2), std=(3, 2, 1))
    >>> image = mx.nd.random.uniform(0, 1, (3, 4, 2))
    >>> transformer(image)
    [[[ 0.18293785  0.19761486]
      [ 0.23839645  0.28142193]
      [ 0.20092112  0.28598186]
      [ 0.18162774  0.28241724]]
     [[-0.2881726  -0.18821815]
      [-0.17705294 -0.30780914]
      [-0.2812064  -0.3512327 ]
      [-0.05411351 -0.4716435 ]]
     [[-1.0363373  -1.7273437 ]
      [-1.6165586  -1.5223348 ]
      [-1.208275   -1.1878313 ]
      [-1.4711051  -1.5200229 ]]]
    <NDArray 3x4x2 @cpu(0)>
    """
    def __init__(self, mean=0.0, std=1.0):
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


    Inputs:
        - **data**: input tensor with (Hi x Wi x C) shape.

    Outputs:
        - **out**: output tensor with (H x W x C) shape.
    """
    def __init__(self, size, scale=(0.08, 1.0), ratio=(3.0/4.0, 4.0/3.0),
                 interpolation=1):
        super(RandomResizedCrop, self).__init__()
        if isinstance(size, numeric_types):
            size = (size, size)
        self._args = (size, scale, ratio, interpolation)

    def forward(self, x):
        return image.random_size_crop(x, *self._args)[0]


class CropResize(HybridBlock):
    r"""Crop the input image with and optionally resize it.

    Makes a crop of the original image then optionally resize it to the specified size.

    Parameters
    ----------
    x : int
        Left boundary of the cropping area
    y : int
        Top boundary of the cropping area
    w : int
        Width of the cropping area
    h : int
        Height of the cropping area
    size : int or tuple of (w, h)
        Optional, resize to new size after cropping
    interpolation : int, optional
        Interpolation method for resizing. By default uses bilinear
        interpolation. See OpenCV's resize function for available choices.
        https://docs.opencv.org/2.4/modules/imgproc/doc/geometric_transformations.html?highlight=resize#resize
        Note that the Resize on gpu use contrib.bilinearResize2D operator
        which only support bilinear interpolation(1). The result would be slightly
        different on gpu compared to cpu. OpenCV tend to align center while bilinearResize2D
        use algorithm which aligns corner.


    Inputs:
        - **data**: input tensor with (H x W x C) or (N x H x W x C) shape.

    Outputs:
        - **out**: input tensor with (H x W x C) or (N x H x W x C) shape.

    Examples
    --------
    >>> transformer = vision.transforms.CropResize(x=0, y=0, width=100, height=100)
    >>> image = mx.nd.random.uniform(0, 255, (224, 224, 3)).astype(dtype=np.uint8)
    >>> transformer(image)
    <NDArray 100x100x3 @cpu(0)>
    >>> image = mx.nd.random.uniform(0, 255, (3, 224, 224, 3)).astype(dtype=np.uint8)
    >>> transformer(image)
    <NDArray 3x100x100x3 @cpu(0)>
    >>> transformer = vision.transforms.CropResize(x=0, y=0, width=100, height=100, size=(50, 50), interpolation=1)
    >>> transformer(image)
    <NDArray 3x50x50 @cpu(0)>
    """
    def __init__(self, x, y, width, height, size=None, interpolation=None):
        super(CropResize, self).__init__()
        self._x = x
        self._y = y
        self._width = width
        self._height = height
        self._size = size
        self._interpolation = interpolation

    def hybrid_forward(self, F, x):
        out = F.image.crop(x, self._x, self._y, self._width, self._height)
        if self._size:
            out = F.image.resize(out, self._size, False, self._interpolation)
        return out

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


    Inputs:
        - **data**: input tensor with (Hi x Wi x C) shape.

    Outputs:
        - **out**: output tensor with (H x W x C) shape.

    Examples
    --------
    >>> transformer = vision.transforms.CenterCrop(size=(1000, 500))
    >>> image = mx.nd.random.uniform(0, 255, (2321, 3482, 3)).astype(dtype=np.uint8)
    >>> transformer(image)
    <NDArray 500x1000x3 @cpu(0)>
    """
    def __init__(self, size, interpolation=1):
        super(CenterCrop, self).__init__()
        if isinstance(size, numeric_types):
            size = (size, size)
        self._args = (size, interpolation)

    def forward(self, x):
        return image.center_crop(x, *self._args)[0]


class Resize(HybridBlock):
    """Resize an image or a batch of image NDArray to the given size.
    Should be applied before `mxnet.gluon.data.vision.transforms.ToTensor`.

    Parameters
    ----------
    size : int or tuple of (W, H)
        Size of output image.
    keep_ratio : bool
        Whether to resize the short edge or both edges to `size`,
        if size is give as an integer.
    interpolation : int
        Interpolation method for resizing. By default uses bilinear
        interpolation. See OpenCV's resize function for available choices.
        Note that the Resize on gpu use contrib.bilinearResize2D operator
        which only support bilinear interpolation(1). The result would be slightly
        different on gpu compared to cpu. OpenCV tend to align center while bilinearResize2D
        use algorithm which aligns corner.


    Inputs:
        - **data**: input tensor with (H x W x C) or (N x H x W x C) shape.

    Outputs:
        - **out**: output tensor with (H x W x C) or (N x H x W x C) shape.

    Examples
    --------
    >>> transformer = vision.transforms.Resize(size=(1000, 500))
    >>> image = mx.nd.random.uniform(0, 255, (224, 224, 3)).astype(dtype=np.uint8)
    >>> transformer(image)
    <NDArray 500x1000x3 @cpu(0)>
    >>> image = mx.nd.random.uniform(0, 255, (3, 224, 224, 3)).astype(dtype=np.uint8)
    >>> transformer(image)
    <NDArray 3x500x1000x3 @cpu(0)>
    """
    def __init__(self, size, keep_ratio=False, interpolation=1):
        super(Resize, self).__init__()
        self._keep = keep_ratio
        self._size = size
        self._interpolation = interpolation

    def hybrid_forward(self, F, x):
        return F.image.resize(x, self._size, self._keep, self._interpolation)

class RandomFlipLeftRight(HybridBlock):
    """Randomly flip the input image left to right with a probability
    of 0.5.

    Inputs:
        - **data**: input tensor with (H x W x C) shape.

    Outputs:
        - **out**: output tensor with same shape as `data`.
    """
    def __init__(self):
        super(RandomFlipLeftRight, self).__init__()

    def hybrid_forward(self, F, x):
        return F.image.random_flip_left_right(x)


class RandomFlipTopBottom(HybridBlock):
    """Randomly flip the input image top to bottom with a probability
    of 0.5.

    Inputs:
        - **data**: input tensor with (H x W x C) shape.

    Outputs:
        - **out**: output tensor with same shape as `data`.
    """
    def __init__(self):
        super(RandomFlipTopBottom, self).__init__()

    def hybrid_forward(self, F, x):
        return F.image.random_flip_top_bottom(x)


class RandomBrightness(HybridBlock):
    """Randomly jitters image brightness with a factor
    chosen from `[max(0, 1 - brightness), 1 + brightness]`.

    Parameters
    ----------
    brightness: float
        How much to jitter brightness. brightness factor is randomly
        chosen from `[max(0, 1 - brightness), 1 + brightness]`.


    Inputs:
        - **data**: input tensor with (H x W x C) shape.

    Outputs:
        - **out**: output tensor with same shape as `data`.
    """
    def __init__(self, brightness):
        super(RandomBrightness, self).__init__()
        self._args = (max(0, 1-brightness), 1+brightness)

    def hybrid_forward(self, F, x):
        return F.image.random_brightness(x, *self._args)


class RandomContrast(HybridBlock):
    """Randomly jitters image contrast with a factor
    chosen from `[max(0, 1 - contrast), 1 + contrast]`.

    Parameters
    ----------
    contrast: float
        How much to jitter contrast. contrast factor is randomly
        chosen from `[max(0, 1 - contrast), 1 + contrast]`.


    Inputs:
        - **data**: input tensor with (H x W x C) shape.

    Outputs:
        - **out**: output tensor with same shape as `data`.
    """
    def __init__(self, contrast):
        super(RandomContrast, self).__init__()
        self._args = (max(0, 1-contrast), 1+contrast)

    def hybrid_forward(self, F, x):
        return F.image.random_contrast(x, *self._args)


class RandomSaturation(HybridBlock):
    """Randomly jitters image saturation with a factor
    chosen from `[max(0, 1 - saturation), 1 + saturation]`.

    Parameters
    ----------
    saturation: float
        How much to jitter saturation. saturation factor is randomly
        chosen from `[max(0, 1 - saturation), 1 + saturation]`.


    Inputs:
        - **data**: input tensor with (H x W x C) shape.

    Outputs:
        - **out**: output tensor with same shape as `data`.
    """
    def __init__(self, saturation):
        super(RandomSaturation, self).__init__()
        self._args = (max(0, 1-saturation), 1+saturation)

    def hybrid_forward(self, F, x):
        return F.image.random_saturation(x, *self._args)


class RandomHue(HybridBlock):
    """Randomly jitters image hue with a factor
    chosen from `[max(0, 1 - hue), 1 + hue]`.

    Parameters
    ----------
    hue: float
        How much to jitter hue. hue factor is randomly
        chosen from `[max(0, 1 - hue), 1 + hue]`.


    Inputs:
        - **data**: input tensor with (H x W x C) shape.

    Outputs:
        - **out**: output tensor with same shape as `data`.
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


    Inputs:
        - **data**: input tensor with (H x W x C) shape.

    Outputs:
        - **out**: output tensor with same shape as `data`.
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


    Inputs:
        - **data**: input tensor with (H x W x C) shape.

    Outputs:
        - **out**: output tensor with same shape as `data`.
    """
    def __init__(self, alpha):
        super(RandomLighting, self).__init__()
        self._alpha = alpha

    def hybrid_forward(self, F, x):
        return F.image.random_lighting(x, self._alpha)
