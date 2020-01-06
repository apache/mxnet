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

import numpy as np
import cv2


def get_image(roi_rec, short, max_size, mean, std):
    """
    read, resize, transform image, return im_tensor, im_info, gt_boxes
    roi_rec should have keys: ["image", "boxes", "gt_classes", "flipped"]
    0 --- x (width, second dim of im)
    |
    y (height, first dim of im)
    """
    im = imdecode(roi_rec['image'])
    if roi_rec["flipped"]:
        im = im[:, ::-1, :]
    im, im_scale = resize(im, short, max_size)
    height, width = im.shape[:2]
    im_info = np.array([height, width, im_scale], dtype=np.float32)
    im_tensor = transform(im, mean, std)

    # gt boxes: (x1, y1, x2, y2, cls)
    if roi_rec['gt_classes'].size > 0:
        gt_inds = np.where(roi_rec['gt_classes'] != 0)[0]
        gt_boxes = np.empty((len(gt_inds), 5), dtype=np.float32)
        gt_boxes[:, 0:4] = roi_rec['boxes'][gt_inds, :]
        gt_boxes[:, 4] = roi_rec['gt_classes'][gt_inds]
        # scale gt_boxes
        gt_boxes[:, 0:4] *= im_scale
    else:
        gt_boxes = np.empty((0, 5), dtype=np.float32)

    return im_tensor, im_info, gt_boxes


def imdecode(image_path):
    """Return BGR image read by opencv"""
    import os
    assert os.path.exists(image_path), image_path + ' not found'
    im = cv2.imread(image_path)
    return im


def resize(im, short, max_size):
    """
    only resize input image to target size and return scale
    :param im: BGR image input by opencv
    :param short: one dimensional size (the short side)
    :param max_size: one dimensional max size (the long side)
    :return: resized image (NDArray) and scale (float)
    """
    im_shape = im.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])
    im_scale = float(short) / float(im_size_min)
    # prevent bigger axis from being more than max_size:
    if np.round(im_scale * im_size_max) > max_size:
        im_scale = float(max_size) / float(im_size_max)
    im = cv2.resize(im, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR)
    return im, im_scale


def transform(im, mean, std):
    """
    transform into mxnet tensor,
    subtract pixel size and transform to correct format
    :param im: [height, width, channel] in BGR
    :param mean: [RGB pixel mean]
    :param std: [RGB pixel std var]
    :return: [batch, channel, height, width]
    """
    im_tensor = np.zeros((3, im.shape[0], im.shape[1]))
    for i in range(3):
        im_tensor[i, :, :] = (im[:, :, 2 - i] - mean[i]) / std[i]
    return im_tensor


def transform_inverse(im_tensor, mean, std):
    """
    transform from mxnet im_tensor to ordinary RGB image
    im_tensor is limited to one image
    :param im_tensor: [batch, channel, height, width]
    :param mean: [RGB pixel mean]
    :param std: [RGB pixel std var]
    :return: im [height, width, channel(RGB)]
    """
    assert im_tensor.shape[0] == 3
    im = im_tensor.transpose((1, 2, 0))
    im = im * std + mean
    im = im.astype(np.uint8)
    return im


def tensor_vstack(tensor_list, pad=0):
    """
    vertically stack tensors by adding a new axis
    expand dims if only 1 tensor
    :param tensor_list: list of tensor to be stacked vertically
    :param pad: label to pad with
    :return: tensor with max shape
    """
    if len(tensor_list) == 1:
        return tensor_list[0][np.newaxis, :]

    ndim = len(tensor_list[0].shape)
    dimensions = [len(tensor_list)]  # first dim is batch size
    for dim in range(ndim):
        dimensions.append(max([tensor.shape[dim] for tensor in tensor_list]))

    dtype = tensor_list[0].dtype
    if pad == 0:
        all_tensor = np.zeros(tuple(dimensions), dtype=dtype)
    elif pad == 1:
        all_tensor = np.ones(tuple(dimensions), dtype=dtype)
    else:
        all_tensor = np.full(tuple(dimensions), pad, dtype=dtype)
    if ndim == 1:
        for ind, tensor in enumerate(tensor_list):
            all_tensor[ind, :tensor.shape[0]] = tensor
    elif ndim == 2:
        for ind, tensor in enumerate(tensor_list):
            all_tensor[ind, :tensor.shape[0], :tensor.shape[1]] = tensor
    elif ndim == 3:
        for ind, tensor in enumerate(tensor_list):
            all_tensor[ind, :tensor.shape[0], :tensor.shape[1], :tensor.shape[2]] = tensor
    else:
        raise Exception('Sorry, unimplemented.')
    return all_tensor
