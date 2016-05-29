import numpy as np
import cv2


def resize(im, target_size, max_size):
    """
    only resize input image to target size and return scale
    :param im: BGR image input by opencv
    :param target_size: one dimensional size (the short side)
    :param max_size: one dimensional max size (the long side)
    :return:
    """
    im_shape = im.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.min(im_shape[0:2])
    im_scale = float(target_size) / float(im_size_min)
    # prevent bigger axis from being more than max_size:
    if np.round(im_scale * im_size_max) > max_size:
        im_scale = float(max_size) / float(im_size_max)
    im = cv2.resize(im, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR)
    return im, im_scale


def transform(im, pixel_means):
    """
    transform into mxnet tensor
    substract pixel size and transform to correct format
    :param im: [height, width, channel] in BGR
    :param pixel_means: [[[R, G, B pixel means]]]
    :return: [batch, channel, height, width]
    """
    im = im.copy()
    im[:, :, (0, 1, 2)] = im[:, :, (2, 1, 0)]
    im = im.astype(float)
    im -= pixel_means
    im_tensor = im[np.newaxis, :]
    # put channel first
    channel_swap = (0, 3, 1, 2)
    im_tensor = im_tensor.transpose(channel_swap)
    return im_tensor


def transform_inverse(im_tensor, pixel_means):
    """
    transform from mxnet im_tensor to ordinary RGB image
    im_tensor is limited to one image
    :param im_tensor: [batch, channel, height, width]
    :param pixel_means: [[[R, G, B pixel means]]]
    :return: im [height, width, channel(RGB)]
    """
    assert im_tensor.shape[0] == 1
    im_tensor = im_tensor.copy()
    # put channel back
    channel_swap = (0, 2, 3, 1)
    im_tensor = im_tensor.transpose(channel_swap)
    im = im_tensor[0]
    assert im.shape[2] == 3
    im += pixel_means
    im = im.astype(np.uint8)
    return im


def tensor_vstack(im_list):
    """
    stack input image (usually 2 image) to obtain input to CNN
    extra regions are padded with zero
    :param im_list: list of image to be stacked vertically
    :return: im_tensor [batch, channel, height, width]
    """
    max_channel = max([im_tensor.shape[1] for im_tensor in im_list])
    max_height = max([im_tensor.shape[2] for im_tensor in im_list])
    max_width = max([im_tensor.shape[3] for im_tensor in im_list])
    im_tensor = np.zeros((len(im_list), max_channel, max_height, max_width))
    for ind, im in enumerate(im_list):
        im_tensor[ind, :im.shape[1], :im.shape[2], :im.shape[3]] = im
    return im_tensor
