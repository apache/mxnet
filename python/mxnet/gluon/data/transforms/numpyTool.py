from __future__ import division

import math
import random
from PIL import Image, ImageOps
import PIL

try:
    import accimage
except ImportError:
    accimage = None

import numpy as np
import numbers
import types
import collections


class ToNumpy(object):
    def __call__(self, pic):

        if not isinstance(pic, PIL.Image.Image):
            raise TypeError('Only support PIL image, type %s is invalid' % type(pic))
            # handle PIL Image

        if pic.mode == 'I':
            img = np.array(pic, np.int32, copy=False)
        elif pic.mode == 'I;16':
            img = np.array(pic, np.int16, copy=False)
        else:
            # img = pic.tobytes()
            img = np.array(pic)

        # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
        if pic.mode == 'YCbCr':
            nchannel = 3
        elif pic.mode == 'I;16':
            nchannel = 1
        else:
            nchannel = len(pic.mode)
        img = img.reshape(pic.size[1], pic.size[0], nchannel)

        # put it from HWC to CHW format
        # yikes, this transpose takes 80% of the loading time/CPU
        # img = img.transpose(0, 1).transpose(0, 2).contiguous()


        # if isinstance(img, torch.ByteTensor):
        #     raise NotImplementedError('Byte Image is not supported yet')
        #     # return img.float().div(255)
        # else:
        #     return img
        return img


class Normalize(object):
    """Normalize an tensor image with mean and standard deviation.

    Given mean: (R, G, B) and std: (R, G, B),
    will normalize each channel of the torch.*Tensor, i.e.
    channel = (channel - mean) / std

    Parameters
    ----------
    mean (sequence): Sequence of means for R, G, B channels respecitvely.
    std (sequence): Sequence of standard deviations for R, G, B channels
        respecitvely.
    """

    def __init__(self, mean, std=[1, 1, 1], axis=0):
        self.mean = mean
        self.std = std
        self.axis = axis

    def __call__(self, tensor):
        """
        Parameters
        ----------
        tensor (Tensor): Tensor image of size (C, H, W) to be normalized.

        """
        # TODO: make efficient
        # for t, m, s in zip(tensor, self.mean, self.std):
        #     t.sub_(m).div_(s)
        for t, m, s in zip(tensor, self.mean, self.std):
            t.__isub__(m).__idiv__(s)
        return tensor
