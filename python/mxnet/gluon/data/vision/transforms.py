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

from .. import dataset
from ...block import Block, HybridBlock
from ...nn import Sequential, HybridSequential
from .... import ndarray, initializer
from ....base import _Null


class Compose(Sequential):
    def __init__(self, transforms):
        super(Compose, self).__init__()
        transforms.append(None)
        hybrid = []
        for i in transforms:
            if isinstance(i, HybridBlock):
                hybrid.append(i)
                continue
            elif len(hybrid) == 1:
                self.register_child(hybrid[0])
            elif len(hybrid) > 1:
                hblock = HybridSequential()
                for j in hybrid:
                    hblock.add(j)
                self.register_child(hblock)
            if i is not None:
                self.register_child(i)
        self.hybridize()


class Cast(HybridBlock):
    def __init__(self, dtype='float32'):
        super(Cast, self).__init__()
        self._dtype = dtype

    def hybrid_forward(self, F, x):
        return F.cast(x, self._dtype)


class ToTensor(HybridBlock):
    def __init__(self):
        super(ToTensor, self).__init__()

    def hybrid_forward(self, F, x):
        return F.image.to_tensor(x)


class Normalize(HybridBlock):
    def __init__(self, mean, std):
        super(Normalize, self).__init__()
        self._mean = mean
        self._std = std

    def hybrid_forward(self, F, x):
        return F.image.normalize(x, self._mean, self._std)


class RandomResizedCrop(HybridBlock):
    def __init__(self, size, area=(0.08, 1.0), ratio=(3.0/4.0, 4.0/3.0),
                 interpolation=2):
        super(RandomResizedCrop, self).__init__()
        self._args = (size, area, ratio, interpolation)

    def hybrid_forward(self, F, x):
        return F.image.random_resized_crop(x, *self._args)


class CenterCrop(HybridBlock):
    def __init__(self, size):
        super(CenterCrop, self).__init__()
        self._size = size

    def hybrid_forward(self, F, x):
        return F.image.center_crop(x, size)


class Resize(HybridBlock):
    def __init__(self, size, interpolation=2):
        super(Resize, self).__init__()
        self._args = (size, interpolation)

    def hybrid_forward(self, F, x):
        return F.image.resize(x, *self._args)


class RandomFlip(HybridBlock):
    def __init__(self, axis=1):
        super(RandomFlip, self).__init__()
        self._axis = axis

    def hybrid_forward(self, F, x):
        return F.image.random_flip(x, self._axis)


class RandomBrightness(HybridBlock):
    def __init__(self, max_brightness):
        super(RandomBrightness, self).__init__()
        self._max_brightness = max_brightness

    def hybrid_forward(self, F, x):
        return F.image.random_brightness(x, self._max_brightness)


class RandomContrast(HybridBlock):
    def __init__(self, max_contrast):
        super(RandomContrast, self).__init__()
        self._max_contrast = max_contrast

    def hybrid_forward(self, F, x):
        return F.image.random_contrast(x, self._max_contrast)


class RandomSaturation(HybridBlock):
    def __init__(self, max_saturation):
        super(RandomSaturation, self).__init__()
        self._max_saturation = max_saturation

    def hybrid_forward(self, F, x):
        return F.image.random_saturation(x, self._max_saturation)


class RandomHue(HybridBlock):
    def __init__(self, max_hue):
        super(RandomHue, self).__init__()
        self._max_hue = max_hue

    def hybrid_forward(self, F, x):
        return F.image.random_hue(x, self._max_hue)


class RandomColorJitter(HybridBlock):
    def __init__(self, max_brightness=0, max_contrast=0, max_saturation=0, max_hue=0):
        super(RandomColorJitter, self).__init__()
        self._args = (max_brightness, max_contrast, max_saturation, max_hue)

    def hybrid_forward(self, F, x):
        return F.image.random_color_jitter(x, *self._args)


class AdjustLighting(HybridBlock):
    def __init__(self, alpha_rgb=_Null, eigval=_Null, eigvec=_Null):
        super(AdjustLighting, self).__init__()
        self._args = (alpha_rgb, eigval, eigvec)

    def hybrid_forward(self, F, x):
        return F.image.adjust_lighting(x, *self._args)


class RandomLighting(HybridBlock):
    def __init__(self, alpha_std=_Null, eigval=_Null, eigvec=_Null):
        super(RandomLighting, self).__init__()
        self._args = (alpha_std, eigval, eigvec)

    def hybrid_forward(self, F, x):
        return F.image.random_lighting(x, *self._args)