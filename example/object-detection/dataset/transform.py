"""Transform functions for data."""
from mxnet import ndarray as nd
from mxnet import image
import numpy as np
import random


class Compose(object):
    """Compose augmentations together.

    Parameters
    -----------

    """
    def __init__(self, transforms):
        self._transforms = transforms

    def __call__(self, src, label):
        for t in self._transforms:
            src, label = t(src, label)
        return src, label

class Lambda(object):
    """Applies lambda function a transform.

    Parameters
    ----------
    func : callable
        A callable function that will do::

            src, label = func(src, label)

    Returns
    -------
    src : NDArray
        Image
    label : numpy.ndarray
        Label
    """
    def __init__(self, func):
        assert callable(lambd), "Lambda function must be callable"
        self._lambda = func

    def __call__(self, src, label):
        return self._lambda(src, label)


class Cast(object):
    """Cast image to another type.

    """
    def __init__(self, typ=np.float32):
        self._type = typ

    def __call__(self, src, label):
        return src.astype(self._type), label.astype(self._type)


class ToAbsoluteCoords(object):
    """Convert box coordinate to pixel values.


    """
    def __call__(self, src, label):
        height, width, _ = src.shape
        label[:, (1, 3)] *= width
        label[:, (2, 4)] *= height
        return src, label

class ToPercentCoords(object):
    """Convert box coordinates to relative percentage values.

    """
    def __call__(self, src, label):
        height, width, _ = src.shape
        label[:, (1, 3)] /= width
        label[:, (2, 4)] /= height
        return src, label


class ForceResize(object):
    """Force resize to data_shape for batch training. Note that coordinates must
    be converted to percent before this augmentation.

    Parameters
    ----------
    size : tuple
        A tuple of (width, height) to be resized to.

    """
    def __init__(self, size):
        self._size = size

    def __call__(self, src, label):
        src = image.imresize(src, *self._size, interp=1)
        return src, label


def intersect(box_a, box_b):
    max_xy = np.minimum(box_a[:, 2:], box_b[2:])
    min_xy = np.maximum(box_a[:, :2], box_b[:2])
    inter = np.clip((max_xy - min_xy), a_min=0, a_max=np.inf)
    return inter[:, 0] * inter[:, 1]

def jaccard_numpy(box_a, box_b):
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.
    E.g.:
    Args:
        box_a: Multiple bounding boxes, Shape: [num_boxes,4]
        box_b: Single bounding box, Shape: [4]
    Return:
        jaccard overlap: Shape: [box_a.shape[0], box_a.shape[1]]
    """
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2]-box_a[:, 0]) *
              (box_a[:, 3]-box_a[:, 1]))  # [A,B]
    area_b = ((box_b[2]-box_b[0]) *
              (box_b[3]-box_b[1]))  # [A,B]
    union = area_a + area_b - inter
    return inter / union # [A,B]


class RandomSampleCrop(object):
    """Randomly crop images and modify labels according to constraints.

    Parameters
    ----------

    """
    def __init__(self, max_attempts=50):
        self._options = (
            # using entire original image
            None,
            # sample a patch s.t. minimum iou
            (0.1, None),
            (0.3, None),
            (0.5, None),
            (0.7, None),
            (0.9, None),
            # randomly sample a patch
            (None, None),
        )
        self._max_attempts = max_attempts


    def __call__(self, src, label):
        height, width, _ = src.shape
        while True:
            # randomly choose a crop mode
            mode = random.choice(self._options)
            if mode is None:
                # return the original intact
                return src, label

            min_iou, max_iou = mode
            if min_iou is None:
                min_iou = float('-inf')
            if max_iou is None:
                max_iou = float('inf')

            # max trails
            for _ in range(self._max_attempts):
                current_image = src
                w = random.uniform(0.3 * width , width)
                h = random.uniform(0.3 * height, height)

                # aspect ratio constraint 0.5
                if h / w < 0.5 or h / w > 2:
                    continue

                left = random.uniform(0, width - w)
                top = random.uniform(0, height - h)

                # convert to integer rect
                rect = np.array([int(left), int(top), int(left+w), int(top+h)])

                # calculate iou
                overlap = jaccard_numpy(label[:, 1:5], rect)

                # check min and max iou constraint? if not try again
                if overlap.min() < min_iou and max_iou < overlap.max():
                    continue

                # crop
                current_image = current_image[rect[1]:rect[3], rect[0]:rect[2], :]

                # keep overlap with gt box is center in sampled patch
                centers = (label[:, 1:3] + label[:, 3:5]) / 2.0

                # mask in all gt boxes that above and to the left of centers
                m1 = (rect[0] < centers[:, 0]) * (rect[1] < centers[:, 1])
                # mask in all gt boxes that under ant to the right of centers
                m2 = (rect[2] > centers[:, 0]) * (rect[3] > centers[:, 1])
                mask = m1 * m2

                # check valid masks
                if not mask.any():
                    continue

                # take only matching gt
                current_label = label[mask, :].copy()

                # should we use the box left and top corner of the crop's
                current_label[:, 1:3] = np.maximum(current_label[:, 1:3], rect[:2])
                # adjust to crop
                current_label[:, 1:3] -= rect[:2]
                current_label[:, 3:5] = np.minimum(current_label[:, 3:5], rect[2:])
                current_label[:, 3:5] -= rect[:2]

                return current_image, current_label
        return src, label

class Expand(object):
    """Randomly pad image.


    """
    def __init__(self, mean_pixel):
        self._mean = [127, 127, 127]

    def __call__(self, src, label):
        if random.randint(0, 1):
            return src, label

        height, width, _ = src.shape
        ratio = random.uniform(1, 4)
        left = int(random.uniform(0, width * ratio - width))
        top = int(random.uniform(0, height * ratio - height))
        new_width = int(width * ratio)
        new_height = int(height * ratio)

        # default using mean pixels
        expand_image = nd.repeat(nd.array(self._mean), repeats=new_width * new_height).reshape((new_height, new_width, -1))
        expand_image[top:top + height, left:left + width, :] = src
        new_label = label.copy()
        new_label[:, 1:3] += (left, top)
        new_label[:, 3:5] += (left, top)

        return expand_image, new_label

class Transpose(object):
    """Transpose to tensor order"""
    def __init__(self, order=(2, 0, 1)):
        self._order = order

    def __call__(self, src, label):
        return nd.transpose(src, axes=self._order), label

class SSDAugmentation(object):
    def __init__(self, data_shape, mean_pixel=[123, 117, 104], std_pixel=[58, 57, 58]):
        self._augments = Compose([
            Cast(),
            ToAbsoluteCoords(),
            Expand(mean_pixel),
            RandomSampleCrop(),
            ToPercentCoords(),
            image.det.DetHorizontalFlipAug(0.5),
            ForceResize(data_shape),
            image.det.DetBorrowAug(image.ColorNormalizeAug(mean_pixel, std_pixel)),
            Transpose(),
        ])

    def __call__(self, src, label):
        # print(self._augments(src, label)[1])
        return self._augments(src, label)

class SSDValid(object):
    def __init__(self, data_shape, mean_pixel=[123, 117, 104], std_pixel=[58, 57, 58]):
        self._augments = Compose([
            Cast(),
            ForceResize(data_shape),
            image.det.DetBorrowAug(image.ColorNormalizeAug(mean_pixel, std_pixel)),
            Transpose(),
        ])

    def __call__(self, src, label):
        # print(self._augments(src, label)[1])
        return self._augments(src, label)

class SSDAugmentation2(object):
    def __init__(self, data_shape):
        ag_list = image.det.CreateDetAugmenter([3] + data_shape, rand_crop=0.8, rand_pad=0.8,
            rand_mirror=True, mean=True, std=True)
        ag_list.append(Transpose())
        ag_list.append(Cast())
        self._augments = Compose(ag_list)

    def __call__(self, src, label):
        return self._augments(src, label)
