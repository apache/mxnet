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
# pylint: disable=too-many-arguments,no-member,invalid-name

"""Opencv plugin for mxnet"""
import random
import ctypes
import cv2
import mxnet as mx
from mxnet.base import _LIB
from mxnet.base import mx_uint, NDArrayHandle, check_call

def imdecode(str_img, flag=1):
    """Decode image from str buffer.
    Wrapper for cv2.imdecode that uses mx.nd.NDArray

    Parameters
    ----------
    str_img : str
        str buffer read from image file
    flag : int
        same as flag for cv2.imdecode
    Returns
    -------
    img : NDArray
        decoded image in (width, height, channels)
        with BGR color channel order
    """
    hdl = NDArrayHandle()
    check_call(_LIB.MXCVImdecode(ctypes.c_char_p(str_img),
                                 mx_uint(len(str_img)),
                                 flag, ctypes.byref(hdl)))
    return mx.nd.NDArray(hdl)

def resize(src, size, interpolation=cv2.INTER_LINEAR):
    """Decode image from str buffer.
    Wrapper for cv2.imresize that uses mx.nd.NDArray

    Parameters
    ----------
    src : NDArray
        image in (width, height, channels)
    size : tuple
        target size in (width, height)
    interpolation : int
        same as interpolation for cv2.imresize

    Returns
    -------
    img : NDArray
        resized image
    """
    hdl = NDArrayHandle()
    check_call(_LIB.MXCVResize(src.handle, mx_uint(size[0]), mx_uint(size[1]),
                               interpolation, ctypes.byref(hdl)))
    return mx.nd.NDArray(hdl)

def copyMakeBorder(src, top, bot, left, right, border_type=cv2.BORDER_CONSTANT, value=0):
    """Pad image border
    Wrapper for cv2.copyMakeBorder that uses mx.nd.NDArray

    Parameters
    ----------
    src : NDArray
        Image in (width, height, channels).
        Others are the same with cv2.copyMakeBorder

    Returns
    -------
    img : NDArray
        padded image
    """
    hdl = NDArrayHandle()
    check_call(_LIB.MXCVcopyMakeBorder(src.handle, ctypes.c_int(top), ctypes.c_int(bot),
                                       ctypes.c_int(left), ctypes.c_int(right),
                                       ctypes.c_int(border_type), ctypes.c_double(value),
                                       ctypes.byref(hdl)))
    return mx.nd.NDArray(hdl)


def scale_down(src_size, size):
    """Scale down crop size if it's bigger than image size"""
    w, h = size
    sw, sh = src_size
    if sh < h:
        w, h = float(w*sh)/h, sh
    if sw < w:
        w, h = sw, float(h*sw)/w
    return int(w), int(h)

def fixed_crop(src, x0, y0, w, h, size=None, interpolation=cv2.INTER_CUBIC):
    """Crop src at fixed location, and (optionally) resize it to size"""
    out = mx.nd.crop(src, begin=(y0, x0, 0), end=(y0+h, x0+w, int(src.shape[2])))
    if size is not None and (w, h) != size:
        out = resize(out, size, interpolation=interpolation)
    return out

def random_crop(src, size):
    """Randomly crop src with size. Upsample result if src is smaller than size"""
    h, w, _ = src.shape
    new_w, new_h = scale_down((w, h), size)

    x0 = random.randint(0, w - new_w)
    y0 = random.randint(0, h - new_h)

    out = fixed_crop(src, x0, y0, new_w, new_h, size)
    return out, (x0, y0, new_w, new_h)

def color_normalize(src, mean, std):
    """Normalize src with mean and std"""
    src -= mean
    src /= std
    return src

def random_size_crop(src, size, min_area=0.25, ratio=(3.0/4.0, 4.0/3.0)):
    """Randomly crop src with size. Randomize area and aspect ratio"""
    h, w, _ = src.shape
    area = w*h
    for _ in range(10):
        new_area = random.uniform(min_area, 1.0) * area
        new_ratio = random.uniform(*ratio)
        new_w = int(new_area*new_ratio)
        new_h = int(new_area/new_ratio)

        if random.uniform(0., 1.) < 0.5:
            new_w, new_h = new_h, new_w

        if new_w > w or new_h > h:
            continue

        x0 = random.randint(0, w - new_w)
        y0 = random.randint(0, h - new_h)

        out = fixed_crop(src, x0, y0, new_w, new_h, size)
        return out, (x0, y0, new_w, new_h)

    return random_crop(src, size)

class ImageListIter(mx.io.DataIter):
    """An example image iterator using opencv plugin"""
    def __init__(self, root, flist, batch_size, size, mean=None):
        mx.io.DataIter.__init__(self)
        self.root = root
        self.list = [line.strip() for line in open(flist).readlines()]
        self.cur = 0
        self.batch_size = batch_size
        self.size = size
        if mean is not None:
            self.mean = mx.nd.array(mean)
        else:
            self.mean = None

    def reset(self):
        """Reset iterator position to 0"""
        self.cur = 0

    def next(self):
        """Move iterator position forward"""
        batch = mx.nd.zeros((self.batch_size, self.size[1], self.size[0], 3))
        i = self.cur
        for i in range(self.cur, min(len(self.list), self.cur+self.batch_size)):
            str_img = open(self.root+self.list[i]+'.jpg').read()
            img = imdecode(str_img, 1)
            img, _ = random_crop(img, self.size)
            batch[i - self.cur] = img
        batch = mx.nd.transpose(batch, axes=(0, 3, 1, 2))
        ret = mx.io.DataBatch(data=[batch],
                              label=[],
                              pad=self.batch_size-(i-self.cur),
                              index=None)
        self.cur = i
        return ret




