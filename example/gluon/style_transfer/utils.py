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

import collections
import os
import numbers
from PIL import Image

import numpy as np
import mxnet as mx
import mxnet.ndarray as F


def tensor_load_rgbimage(filename, ctx, size=None, scale=None, keep_asp=False):
    img = Image.open(filename).convert('RGB')
    if size is not None:
        if keep_asp:
            size2 = int(size * 1.0 / img.size[0] * img.size[1])
            img = img.resize((size, size2), Image.ANTIALIAS)
        else:
            img = img.resize((size, size), Image.ANTIALIAS)

    elif scale is not None:
        img = img.resize((int(img.size[0] / scale), int(img.size[1] / scale)), Image.ANTIALIAS)
    img = np.array(img).transpose(2, 0, 1).astype(float)
    img = F.expand_dims(mx.nd.array(img, ctx=ctx), 0)
    return img


def tensor_save_rgbimage(img, filename, cuda=False):
    img = F.clip(img, 0, 255).asnumpy()
    img = img.transpose(1, 2, 0).astype('uint8')
    img = Image.fromarray(img)
    img.save(filename)


def tensor_save_bgrimage(tensor, filename, cuda=False):
    (b, g, r) = F.split(tensor, num_outputs=3, axis=0)
    tensor = F.concat(r, g, b, dim=0)
    tensor_save_rgbimage(tensor, filename, cuda)


def subtract_imagenet_mean_batch(batch):
    """Subtract ImageNet mean pixel-wise from a BGR image."""
    batch = F.swapaxes(batch,0, 1)
    (r, g, b) = F.split(batch, num_outputs=3, axis=0)
    r = r - 123.680
    g = g - 116.779
    b = b - 103.939
    batch = F.concat(r, g, b, dim=0)
    batch = F.swapaxes(batch,0, 1)
    return batch


def subtract_imagenet_mean_preprocess_batch(batch):
    """Subtract ImageNet mean pixel-wise from a BGR image."""
    batch = F.swapaxes(batch,0, 1)
    (r, g, b) = F.split(batch, num_outputs=3, axis=0)
    r = r - 123.680
    g = g - 116.779
    b = b - 103.939
    batch = F.concat(b, g, r, dim=0)
    batch = F.swapaxes(batch,0, 1)
    return batch


def add_imagenet_mean_batch(batch):
    batch = F.swapaxes(batch,0, 1)
    (b, g, r) = F.split(batch, num_outputs=3, axis=0)
    r = r + 123.680
    g = g + 116.779
    b = b + 103.939
    batch = F.concat(b, g, r, dim=0)
    batch = F.swapaxes(batch,0, 1)
    """
    batch = denormalizer(batch)
    """
    return batch


def imagenet_clamp_batch(batch, low, high):
    """ Not necessary in practice """
    F.clip(batch[:,0,:,:],low-123.680, high-123.680)
    F.clip(batch[:,1,:,:],low-116.779, high-116.779)
    F.clip(batch[:,2,:,:],low-103.939, high-103.939)


def preprocess_batch(batch):
    batch = F.swapaxes(batch, 0, 1)
    (r, g, b) = F.split(batch, num_outputs=3, axis=0)
    batch = F.concat(b, g, r, dim=0)
    batch = F.swapaxes(batch, 0, 1)
    return batch


class ToTensor(object):
    def __init__(self, ctx):
        self.ctx = ctx

    def __call__(self, img):
        img = mx.nd.array(np.array(img).transpose(2, 0, 1).astype('float32'), ctx=self.ctx)
        return img


class Compose(object):
    """Composes several transforms together.
    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.
    Example:
        >>> transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img


class Scale(object):
    """Rescale the input PIL.Image to the given size.
    Args:
        size (sequence or int): Desired output size. If size is a sequence like
            (w, h), output size will be matched to this. If size is an int,
            smaller edge of the image will be matched to this number.
            i.e, if height > width, then image will be rescaled to
            (size * height / width, size)
        interpolation (int, optional): Desired interpolation. Default is
            ``PIL.Image.BILINEAR``
    """

    def __init__(self, size, interpolation=Image.BILINEAR):
        assert isinstance(size, int) or (isinstance(size, collections.Iterable) and len(size) == 2)
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img):
        """
        Args:
            img (PIL.Image): Image to be scaled.
        Returns:
            PIL.Image: Rescaled image.
        """
        if isinstance(self.size, int):
            w, h = img.size
            if (w <= h and w == self.size) or (h <= w and h == self.size):
                return img
            if w < h:
                ow = self.size
                oh = int(self.size * h / w)
                return img.resize((ow, oh), self.interpolation)
            else:
                oh = self.size
                ow = int(self.size * w / h)
                return img.resize((ow, oh), self.interpolation)
        else:
            return img.resize(self.size, self.interpolation)


class CenterCrop(object):
    """Crops the given PIL.Image at the center.
    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img):
        """
        Args:
            img (PIL.Image): Image to be cropped.
        Returns:
            PIL.Image: Cropped image.
        """
        w, h = img.size
        th, tw = self.size
        x1 = int(round((w - tw) / 2.))
        y1 = int(round((h - th) / 2.))
        return img.crop((x1, y1, x1 + tw, y1 + th))


class StyleLoader():
    def __init__(self, style_folder, style_size, ctx):
        self.folder = style_folder
        self.style_size = style_size
        self.files = os.listdir(style_folder)
        assert(len(self.files) > 0)
        self.ctx = ctx

    def get(self, i):
        idx = i%len(self.files)
        filepath = os.path.join(self.folder, self.files[idx])
        style = tensor_load_rgbimage(filepath, self.ctx, self.style_size)
        return style

    def size(self):
        return len(self.files)

def init_vgg_params(vgg, model_folder, ctx):
    if not os.path.exists(os.path.join(model_folder, 'mxvgg.params')):
        os.system('wget https://www.dropbox.com/s/7c92s0guekwrwzf/mxvgg.params?dl=1 -O' + os.path.join(model_folder, 'mxvgg.params'))
    vgg.collect_params().load(os.path.join(model_folder, 'mxvgg.params'), ctx=ctx)
    for param in vgg.collect_params().values():
        param.grad_req = 'null'
