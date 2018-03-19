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

"""Util functions for writing summaries."""

import os
import logging
import numpy as np

try:
    from PIL import Image
except ImportError:
    Image = None
from ...ndarray import NDArray
from ...ndarray import ndarray as nd
from ...ndarray import op


def _make_numpy_array(x):
    if isinstance(x, np.ndarray):
        return x
    elif np.isscalar(x):
        return np.array([x])
    elif isinstance(x, NDArray):
        return x.asnumpy()
    else:
        raise TypeError('_make_numpy_array only accepts input types of numpy.ndarray, scalar,'
                        ' and MXNet NDArray, while received type {}'.format(str(type(x))))


def make_image_grid(tensor, nrow=8, padding=2, normalize=False, norm_range=None,
                    scale_each=False, pad_value=0):
    """Make a grid of images. This is an MXNet version of torchvision.utils.make_grid
    Ref: https://github.com/pytorch/vision/blob/master/torchvision/utils.py

    Parameters
    ----------
        tensor : `NDArray` or list of `NDArray`s
            Input image(s) in the format of HW, CHW, or NCHW.
        nrow : int
            Number of images displayed in each row of the grid. The Final grid size is
            (batch_size / `nrow`, `nrow`).
        padding : int
            Padding value for each image in the grid.
        normalize : bool
            If True, shift the image to the range (0, 1), by subtracting the
            minimum and dividing by the maximum pixel value.
        norm_range : tuple
            Tuple of (min, max) where min and max are numbers. These numbers are used
            to normalize the image. By default, `min` and `max` are computed from the `tensor`.
        scale_each : bool
            If True, scale each image in the batch of images separately rather than the
            `(min, max)` over all images.
        pad_value : float
            Value for the padded pixels.

    Returns
    -------
    NDArray
        A image grid made of the input images.
    """
    if not isinstance(tensor, NDArray) or not (isinstance(tensor, NDArray) and
                                               all(isinstance(t, NDArray) for t in tensor)):
        raise TypeError('MXNet NDArray or list of NDArrays expected, got {}'.format(
            str(type(tensor))))

    # if list of tensors, convert to a 4D mini-batch Tensor
    if isinstance(tensor, list):
        tensor = op.stack(tensor, axis=0)

    if tensor.ndim <= 1 or tensor.ndim > 4:
        raise ValueError('expected 2D, 3D, or 4D NDArrays, while received ndim={}'.format(
            tensor.ndim))

    if tensor.ndim == 2:  # single image H x W
        tensor = tensor.reshape(((1,) + tensor.shape))
    if tensor.ndim == 3:  # single image
        if tensor.shape[0] == 1:  # if single-channel, convert to 3-channel
            tensor = op.concat(*(tensor, tensor, tensor), dim=0)
        tensor = tensor.reshape((1,) + tensor.shape)
    if tensor.ndim == 4 and tensor.shape[1] == 1:  # single-channel images
        tensor = op.concat(*(tensor, tensor, tensor), dim=1)

    if normalize is True:
        tensor = tensor.copy()  # avoid modifying tensor in-place
        if norm_range is not None:
            assert isinstance(norm_range, tuple) and len(norm_range) == 2, \
                "norm_range has to be a tuple (min, max) if specified. min and max are numbers"

        def norm_ip(img, val_min, val_max):
            op.clip(img, a_min=val_min, a_max=val_max, out=img)
            img -= val_min
            img /= (val_max - val_min)

        def norm_range_helper(t, val_range):
            if val_range is not None:
                norm_ip(t, val_range[0], val_range[1])
            else:
                norm_ip(t, t.min(), t.max())

        if scale_each is True:
            for t in tensor:  # loop over mini-batch dimension
                norm_range_helper(t, norm_range)
        else:
            norm_range_helper(tensor, norm_range)

    # if single image, just return
    if tensor.shape[0] == 1:
        return tensor.squeeze(axis=0)

    # make the batch of images into a grid
    nmaps = tensor.shape[0]
    xmaps = min(nrow, nmaps)
    ymaps = int(np.ceil(float(nmaps) / xmaps))
    height, width = int(tensor.shape[2] + padding), int(tensor.shape[3] + padding)
    grid = nd.empty(shape=(3, height * ymaps + padding, width * xmaps + padding),
                    dtype=tensor.dtype, ctx=tensor.context)
    grid[:] = pad_value
    k = 0
    for y in range(ymaps):
        for x in range(xmaps):
            if k >= nmaps:
                break
            start1 = y * height + padding
            end1 = start1 + height - padding
            start2 = x * width + padding
            end2 = start2 + width - padding
            grid[:, start1:end1, start2:end2] = tensor[k]
            k = k + 1
    return grid


def _save_image(image, filename, nrow=8, padding=2):
    """Saves a given Tensor into an image file. If the input tensor contains multiple images,
    a grid of images will be saved.

    Parameters
    ----------
        image : `NDArray`
            Input image(s) in the format of HW, CHW, or NCHW.
        filename : str
            Filename of the saved image(s).
        nrow : int
            Number of images displayed in each row of the grid. The Final grid size is
            (batch_size / `nrow`, `nrow`).
        padding : int
            Padding value for each image in the grid.
    """
    if not isinstance(image, NDArray):
        raise TypeError('MXNet NDArray expected, received {}'.format(str(type(image))))
    image = _prepare_image(image, nrow=nrow, padding=padding)
    if Image is None:
        raise ImportError('saving image failed because PIL is not found')
    im = Image.fromarray(image.asnumpy())
    im.save(filename)


def _prepare_image(img, nrow=8, padding=2):
    """Given an image of format HW, CHW, or NCHW, returns a image of format HWC.
    If the input is a batch of images, a grid of images is made by stitching them together.
    For float input data types, the values are normalized one image at a time to fit in the range
    `[0, 255]`. 'uint8` values are unchanged. The following two normalization algorithms are used
    for different conditions:
    1. If the input values are all positive, they are rescaled so that the largest one is 255.
    2. If any input value is negative, the values are shifted so that the input value 0.0 is at 127.
    They are then rescaled so that either the smallest value is 0, or the largest one is 255.
    This logic is adapted from the `image()` function in
    https://github.com/tensorflow/tensorflow/blob/r1.6/tensorflow/python/summary/summary.py
    It returns an image with as `NDArray` with the color channel in the end of the dimensions.
    """
    assert img.ndim == 2 or img.ndim == 3 or img.ndim == 4
    if isinstance(img, NDArray):
        if img.dtype == np.uint8:
            return make_image_grid(img, nrow=nrow, padding=padding).transpose((1, 2, 0))
        elif img.dtype == np.float16 or img.dtype == np.float32 or img.dtype == np.float64:
            min_val = img.min().asscalar()
            max_val = img.max().asscalar()
            if min_val >= 0:
                min_val = 0.0
            else:
                min_val += 127.0
                max_val += 127.0
                img = img + 127.0
            return (make_image_grid(img, nrow=nrow, padding=padding, normalize=True,
                                    norm_range=(min_val, max_val),
                                    scale_each=True) * 255.0).astype(np.uint8).transpose((1, 2, 0))
        else:
            raise ValueError('expected input image dtype is one of uint8, float16, float32, '
                             'and float64, received dtype {}'.format(str(img.dtype)))
    else:
        raise TypeError('expected MXNet NDArray, while received type {}'.format(str(type(img))))


def _make_metadata_tsv(metadata, save_path):
    """Given an `NDArray` or a `numpy.ndarray` as metadata e.g. labels, save the flattened array
    into the file metadata.tsv under the path provided by the user. Made to satisfy the requirement
    in the following link:
    https://www.tensorflow.org/programmers_guide/embedding#metadata"""
    if isinstance(metadata, NDArray):
        metadata = metadata.asnumpy().flatten()
    elif isinstance(metadata, np.ndarray):
        metadata = metadata.flatten()
    else:
        raise TypeError('expected NDArray of np.ndarray, while received '
                        'type {}'.format(str(type(metadata))))
    metadata = [str(x) for x in metadata]
    with open(os.path.join(save_path, 'metadata.tsv'), 'w') as f:
        for x in metadata:
            f.write(x + '\n')


def _make_sprite_image(images, save_path):
    """Given an NDArray as a batch images, make a sprite image out of it following the rule
    defined in
    https://www.tensorflow.org/programmers_guide/embedding
    and save it in sprite.png under the path provided by the user."""
    assert isinstance(images, NDArray)
    shape = images.shape
    nrow = int(np.ceil(np.sqrt(shape[0])))
    _save_image(images, os.path.join(save_path, 'sprite.png'), nrow=nrow, padding=0)


def _add_embedding_config(file_path, global_step, has_metadata=False,
                          label_img_shape=None, tag='default'):
    """Creates a config file used by the embedding projector.
    Adapted from the TensorFlow function `visualize_embeddings()` at
    https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/tensorboard/plugins/projector/__init__.py"""
    with open(os.path.join(file_path, 'projector_config.pbtxt'), 'a') as f:
        s = 'embeddings {\n'
        s += 'tensor_name: "{}:{}"\n'.format(tag, global_step)
        s += 'tensor_path: "{}"\n'.format(os.path.join(global_step, 'tensors.tsv'))
        if has_metadata:
            s += 'metadata_path: "{}"\n'.format(os.path.join(global_step, 'metadata.tsv'))
        if label_img_shape is not None:
            if len(label_img_shape) != 4:
                logging.warn('expected 4D sprite image in the format NCHW, while received image '
                             'ndim={}, skipping saving sprite '
                             'image info'.format(len(label_img_shape)))
            else:
                s += 'sprite {\n'
                s += 'image_path: "{}"\n'.format(os.path.join(global_step, 'sprite.png'))
                s += 'single_image_dim: {}\n'.format(label_img_shape[3])
                s += 'single_image_dim: {}\n'.format(label_img_shape[2])
                s += '}\n'
        s += '}\n'
        f.write(s)


def _save_embedding_tsv(data, file_path):
    """Given a 2D `NDarray` or a `numpy.ndarray` as embeding,
    save it in tensors.tsv under the path provided by the user."""
    if isinstance(data, np.ndarray):
        data_list = data.tolist()
    elif isinstance(data, NDArray):
        data_list = data.asnumpy().tolist()
    else:
        raise TypeError('expected NDArray of np.ndarray, while received type {}'.format(
            str(type(data))))
    with open(os.path.join(file_path, 'tensors.tsv'), 'w') as f:
        for x in data_list:
            x = [str(i) for i in x]
            f.write('\t'.join(x) + '\n')
