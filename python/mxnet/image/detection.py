# pylint: disable=unused-import
"""Read images and perform augmentations for object detection."""

from __future__ import absolute_import, print_function

import random
import logging
import numpy as np

from ..base import numeric_types
from .. import ndarray as nd
from .._ndarray_internal import _cvcopyMakeBorder as copyMakeBorder
from .. import io
from .image import RandomOrderAug, ColorJitterAug, LightingAug, ColorNormalizeAug
from .image import ResizeAug, ForceResizeAug, CastAug, fixed_crop, ImageIter


def BorrowAug(augmenter):
    """Borrow standard augmenter from image classification
    which is good if label won't be affected."""

    def aug(src, label):
        """Augmenter body"""
        src = augmenter(src)[0]
        return (src, label)

    return aug


def DetHorizontalFlipAug(p):
    """Random horizontal flippling."""

    def horizontal_flip_label(label):
        """Helper function to flip label."""
        tmp = 1.0 - label[:, 3]
        label[:, 1] = tmp
        label[:, 3] = tmp

    def aug(src, label):
        """Augmenter body"""
        if random.random() < p:
            src = nd.flip(src, axis=1)
            # flip ground-truths as well
            horizontal_flip_label(label)
        return (src, label)

    return aug

def DetRandomCropAug(p, min_object_covered=0.1, aspect_ratio_range=(0.75, 1.33),
                     area_range=(0.05, 1.0), max_attempts=50, interp=2):
    """Random cropping with constraints."""
    _enabled = True
    if not isinstance(aspect_ratio_range, tuple):
        assert isinstance(aspect_ratio_range, numeric_types)
        logging.warn('Using fixed aspect ratio: %s', str(aspect_ratio_range))
        aspect_ratio_range = (aspect_ratio_range, aspect_ratio_range)
    if not isinstance(area_range, tuple):
        assert isinstance(area_range, numeric_types)
        logging.warn('Using fixed area: %s', str(area_range))
        area_range = (area_range, area_range)
    if (area_range[1] <= 0 or area_range[0] > area_range[1]):
        logging.warn('Skip random cropping due to invalid parameters.')
        _enabled = False

    def calculate_areas(label):
        """Calculate areas for multiple labels"""
        heights = np.maximum(0, label[:, 3] - label[:, 1])
        widths = np.maximum(0, label[:, 2] - label[:, 0])
        return heights * widths


    def intersect(label, xmin, ymin, xmax, ymax):
        """Calculate intersect areas, normalized."""
        left = np.maximum(label[:, 0], xmin)
        right = np.minimum(label[:, 2], xmax)
        top = np.maximum(label[:, 1], ymin)
        bot = np.minimum(label[:, 3], ymax)
        invalid = np.where(np.logical_or(left >= right, top >= bot))[0]
        out = label.copy()
        out[invalid, :] = 0
        return out

    def check_satisfy_constraints(label, xmin, ymin, xmax, ymax, width, height):
        """Check if constrains are satisfied"""
        if (xmax - xmin) * (ymax - ymin) < 2:
            return False  # only 1 pixel
        x1 = float(xmin) / width
        y1 = float(ymin) / height
        x2 = float(xmax) / width
        y2 = float(ymax) / height
        object_areas = calculate_areas(label[:, 1:])
        valid_objects = np.where(object_areas * width * height < 2)[0]
        intersects = intersect(label[valid_objects, 1:], x1, y1, x2, y2)
        coverages = calculate_areas(intersects) / object_areas
        if np.amin(coverages) > min_object_covered:
            return True

    def update_labels(label, crop_box, height, width):
        """Convert labels according to crop box"""
        xmin = float(crop_box[0]) / width
        ymin = float(crop_box[1]) / height
        w = float(crop_box[2]) / width
        h = float(crop_box[3]) / height
        out = label.copy()
        out[:, (1, 3)] -= xmin
        out[:, (2, 4)] -= ymin
        out = np.maximum(0, out)
        valid = np.where(np.logical_and(out[:, 3] > out[:, 1], out[:, 4] > out[:, 2]))[0]
        if valid.size < 1:
            return None
        out = out[valid, :]
        out[:, (1, 3)] /= w
        out[:, (2, 4)] /= h
        out = np.minimum(1, out)
        return out

    def random_crop_proposal(label, height, width):
        """Propose cropping areas"""
        from math import sqrt

        if not _enabled or height <= 0 or width <= 0:
            return None
        min_area = area_range[0] * height * width
        max_area = area_range[1] * height * width
        for _ in range(max_attempts):
            ratio = random.uniform(*aspect_ratio_range)
            if ratio <= 0:
                continue
            h = int(round(sqrt(min_area / ratio)))
            max_h = int(round(sqrt(max_area / ratio)))
            if round(max_h * ratio) > width:
                # find smallest max_h satifying round(max_h * ratio) <= width
                max_h = int((width + 0.4999999) / ratio)
            if max_h > height:
                max_h = height
            if h > max_h:
                h = max_h
            if h < max_h:
                # generate random h in range [h, max_h]
                h = random.randint(h, max_h)
            w = int(round(h * ratio))
            assert w <= width

            # trying to fix rounding problems
            area = w * h
            if area < min_area:
                h += 1
                w = int(round(h * ratio))
                area = w * h
            if area > max_area:
                h -= 1
                w = int(round(h * ratio))
                area = w * h
            if (area < min_area or area > max_area or w > width or h > height \
                or w <= 0 or h <= 0):
                continue

            y = random.randint(0, max(0, height - h))
            x = random.randint(0, max(0, width - w))
            if check_satisfy_constraints(label, x, y, x + w, y + h, width, height):
                new_label = update_labels(label, (x, y, w, h), height, width)
                if new_label is not None:
                    return (x, y, w, h, new_label)
        return None


    def aug(src, label):
        """Augmenter body"""
        crop = None
        if random.random() < p:
            crop = random_crop_proposal(label, src.shape[0], src.shape[1])
        if crop:
            x, y, w, h, label = crop
            src = fixed_crop(src, x, y, w, h, None, interp)
        return (src, label)

    return aug


def DetRandomPadAug(p, pad_val, aspect_ratio_range=(0.75, 1.33), area_range=(1.0, 3.0),
                    max_attempts=50):
    """Random padding with constraints."""
    _enabled = True
    if not isinstance(aspect_ratio_range, tuple):
        assert isinstance(aspect_ratio_range, numeric_types)
        logging.warn('Using fixed aspect ratio: %.3f', str(aspect_ratio_range))
        aspect_ratio_range = (aspect_ratio_range, aspect_ratio_range)
    if not isinstance(area_range, tuple):
        assert isinstance(area_range, numeric_types)
        area_range = (1.0, area_range)
    if (area_range[1] <= 1.0 or area_range[0] > area_range[1]):
        logging.warn('Skip random padding due to invalid parameters.')
        _enabled = False

    def update_labels(label, pad_box, height, width):
        """Update label according to padding region"""
        out = label.copy()
        out[:, (1, 3)] = (out[:, (1, 3)] * width + pad_box[0]) / pad_box[2]
        out[:, (2, 4)] = (out[:, (2, 4)] * height + pad_box[1]) / pad_box[3]
        return out

    def random_pad_proposal(label, height, width):
        """Generate random padding region"""
        from math import sqrt
        if not _enabled or height <= 0 or width <= 0:
            return None
        min_area = area_range[0] * height * width
        max_area = area_range[1] * height * width
        for _ in range(max_attempts):
            ratio = random.uniform(*aspect_ratio_range)
            if ratio <= 0:
                continue
            h = int(round(sqrt(min_area / ratio)))
            max_h = int(round(sqrt(max_area / ratio)))
            if round(h * ratio) < width:
                h = int((width + 0.499999) / ratio)
            if h < height:
                h = height
            if h > max_h:
                h = max_h
            if h < max_h:
                h = random.randint(h, max_h)
            w = int(round(h * ratio))
            if (h - height) < 2 or (w - width) < 2:
                continue  # marginal padding is not helpful

            y = random.randint(0, max(0, h - height))
            x = random.randint(0, max(0, w - width))
            new_label = update_labels(label, (x, y, w, h), height, width)
            return (x, y, w, h, new_label)
        return None

    def aug(src, label):
        """Augmenter body"""
        pad = None
        height, width, _ = src.shape
        if random.random() < p:
            pad = random_pad_proposal(label, height, width)
        if pad:
            x, y, w, h, label = pad
            src = copyMakeBorder(src, y, h-y-height, x, w-x-width, 16L, pad_val)
        return (src, label)

    return aug


def CreateDetAugmenter(data_shape, resize=0, rand_crop=0, rand_pad=0,
                       rand_mirror=False, mean=None, std=None, brightness=0, contrast=0,
                       saturation=0, pca_noise=0, inter_method=2, min_object_covered=0.1,
                       aspect_ratio_range=(0.75, 1.33), area_range=(0.05, 3.0),
                       max_attempts=50, pad_val=(127, 127, 127)):
    """Create augmenters for detection.

    Parameters
    ----------
    data_shape : tuple of int
        shape for output data
    resize : int
        resize shorter edge if larger than 0 at the begining
    rand_crop : float
        [0.0, 1.0], probability to apply random cropping
    rand_pad : float
        [0.0, 1.0], probability to apply random padding
    rand_mirror : bool
        whether apply horizontal flip to image with probability 0.5
    mean : np.ndarray or None
        mean pixel values for [r, g, b]
    std : np.ndarray or None
        standard deviations for [r, g, b]
    brightness : float
        brightness jittering range (percent)
    contrast : float
        contrast jittering range
    saturation : float
        saturation jittering range
    pca_noise : float
        pca noise level
    inter_method : int
        interpolation method for all resizing operations
        Possible values:
        0: Nearest Neighbors Interpolation.
        1: Bilinear interpolation.
        2: Area-based (resampling using pixel area relation). It may be a
        preferred method for image decimation, as it gives moire-free
        results. But when the image is zoomed, it is similar to the Nearest
        Neighbors method. (used by default).
        3: Bicubic interpolation over 4x4 pixel neighborhood.
        4: Lanczos interpolation over 8x8 pixel neighborhood.
        10: random select from 0 to 4
    min_object_covered : float
        The cropped area of the image must contain at least this fraction of
        any bounding box supplied. The value of this parameter should be non-negative.
        In the case of 0, the cropped area does not need to overlap any of the
        bounding boxes supplied.
    aspect_ratio_range : tuple of floats
        The cropped area of the image must have an aspect ratio = width / height
        within this range.
    area_range : tuple of floats
        The cropped area of the image must contain a fraction of the supplied
        image within in this range.
    max_attempts : int
        Number of attempts at generating a cropped/padded region of the image of the
        specified constraints. After max_attempts failures, return the original image.
    pad_val: float
        pixel value to be filled when padding is enabled. pad_val will automatically
        be subtracted by mean and divided by std if applicable.
    """
    auglist = []

    if inter_method == 10:
        inter_method = random.randint(0, 4)

    if resize > 0:
        auglist.append(BorrowAug(ResizeAug(resize, inter_method)))

    if rand_crop > 0:
        auglist.append(DetRandomCropAug(rand_crop, min_object_covered, aspect_ratio_range,
                                        (area_range[0], 1.0), max_attempts, inter_method))

    if rand_mirror > 0:
        auglist.append(DetHorizontalFlipAug(0.5))

    # apply random padding as late as possible to save computation
    if rand_pad > 0:
        auglist.append(DetRandomPadAug(rand_pad, pad_val, aspect_ratio_range,
                                       (1.0, area_range[1])))

    # force resize
    auglist.append(BorrowAug(ForceResizeAug((data_shape[2], data_shape[1]), inter_method)))

    auglist.append(BorrowAug(CastAug()))

    if brightness or contrast or saturation:
        auglist.append(BorrowAug(ColorJitterAug(brightness, contrast, saturation)))

    if pca_noise > 0:
        eigval = np.array([55.46, 4.794, 1.148])
        eigvec = np.array([[-0.5675, 0.7192, 0.4009],
                           [-0.5808, -0.0045, -0.8140],
                           [-0.5836, -0.6948, 0.4203]])
        auglist.append(BorrowAug(LightingAug(pca_noise, eigval, eigvec)))

    if mean is True:
        mean = np.array([123.68, 116.28, 103.53])
    elif mean is not None:
        assert isinstance(mean, np.ndarray) and mean.shape[0] in [1, 3]

    if std is True:
        std = np.array([58.395, 57.12, 57.375])
    elif std is not None:
        assert isinstance(std, np.ndarray) and std.shape[0] in [1, 3]

    if mean is not None and std is not None:
        auglist.append(BorrowAug(ColorNormalizeAug(mean, std)))

    return auglist


class ImageDetIter(ImageIter):
    """Image iterator with a large number of augmentation choices for detection.

    Parameters
    ----------
    aug_list : list or None
        augmenter list for generating distorted images
    label_name : str
        name for detection labels
    kwargs : ...
        More arguments see mx.image.ImageIter
    """
    def __init__(self, batch_size, data_shape,
                 path_imgrec=None, path_imglist=None, path_root=None, path_imgidx=None,
                 shuffle=False, part_index=0, num_parts=1, aug_list=None, imglist=None,
                 data_name='data', label_name='label', **kwargs):
        super(ImageDetIter, self).__init__(batch_size=batch_size, data_shape=data_shape,
                                           path_imgrec=path_imgrec, path_imglist=path_imglist,
                                           path_root=path_root, path_imgidx=path_imgidx,
                                           shuffle=shuffle, part_index=part_index,
                                           num_parts=num_parts, aug_list=[], imglist=imglist,
                                           data_name=data_name, label_name=label_name)

        if aug_list is None:
            self.auglist = CreateDetAugmenter(data_shape, **kwargs)
        else:
            self.auglist = aug_list

        # went through all labels to get the proper label shape
        label_shape = self._estimate_label_shape()
        self.provide_label = [(label_name, (self.batch_size, label_shape[0], label_shape[1]))]
        self.label_shape = label_shape

    def _check_valid_label(self, label):
        """Validate label and its shape."""
        if len(label.shape) != 2 or label.shape[1] < 5:
            msg = "Label with shape (1+, 5+) required, %s received." % str(label)
            raise RuntimeError(msg)

    def _estimate_label_shape(self):
        """Helper function to estimate label shape"""
        max_count = 0
        self.reset()
        try:
            while True:
                label, _ = self.next_sample()
                label = self._parse_label(label)
                max_count = max(max_count, label.shape[0])
        except StopIteration:
            pass
        self.reset()
        return (max_count, label.shape[1])

    def _parse_label(self, label):
        """Helper function to parse object detection label.

        Format for raw label:
        n \t k \t ... \t [id \t xmin\t ymin \t xmax \t ymax \t ...] \t [repeat]
        where n is the width of header, 2 or larger
        k is the width of each object annotation, can be arbitrary, at least 5
        """
        if isinstance(label, nd.NDArray):
            label = label.asnumpy()
        raw = label.ravel()
        if raw.size < 7:
            raise RuntimeError("Label shape is invalid: " + str(raw.shape))
        header_width = int(raw[0])
        obj_width = int(raw[1])
        if (raw.size - header_width) % obj_width != 0:
            msg = "Label shape %s inconsistent with annotation width %d." \
                %(str(raw.shape, obj_width))
            raise RuntimeError(msg)
        out = np.reshape(raw[header_width:], (-1, obj_width))
        # remove bad ground-truths
        valid = np.where(np.logical_and(out[:, 3] > out[:, 1], out[:, 4] > out[:, 2]))[0]
        if valid.size < 1:
            raise RuntimeError('Encounter sample with no valid label.')
        return out[valid, :]

    def reshape(self, data_shape=None, label_shape=None):
        """Reshape iterator for data_shape or label_shape.

        Parameters
        ----------
        data_shape : tuple or None
            reshape the data_shape to the new shape if not None
        label_shape : tuple or None
            reshape label shape to new shape if not None
        """
        if data_shape is not None:
            self.check_data_shape(data_shape)
            self.provide_data = [(self.provide_data[0][0], (self.batch_size,) + data_shape)]
        if label_shape is not None:
            self.check_label_shape(label_shape)
            self.provide_label = [(self.provide_label[0][0], (self.batch_size,) + label_shape)]

    def next(self):
        """Override the function for returning next batch."""
        batch_size = self.batch_size
        c, h, w = self.data_shape
        batch_data = nd.empty((batch_size, c, h, w))
        batch_label = nd.empty(self.provide_label[0][1])
        i = 0
        try:
            while i < batch_size:
                label, s = self.next_sample()
                data = self.imdecode(s)
                try:
                    self.check_valid_image([data])
                    label = self._parse_label(label)
                except RuntimeError as e:
                    logging.debug('Invalid image, skipping:  %s', str(e))
                    continue
                data, label = self.augmentation_transform(data, label)
                for datum in [data]:
                    assert i < batch_size, 'Batch size must be multiples of augmenter output length'
                    batch_data[i][:] = self.postprocess_data(datum)
                    num_object = label.shape[0]
                    batch_label[i][0:num_object] = nd.array(label)
                    batch_label[i][num_object:] = -1
                    i += 1
        except StopIteration:
            if not i:
                raise StopIteration

        return io.DataBatch([batch_data], [batch_label], batch_size - i)

    def augmentation_transform(self, data, label):
        # pylint: disable=arguments-differ
        """Override Transforms input data with specified augmentations."""
        for aug in self.auglist:
            data, label = aug(data, label)
        return (data, label)

    def check_label_shape(self, label_shape):
        """Checks if the new label shape is valid"""
        if not len(label_shape) == 2:
            raise ValueError('label_shape should have length 2')
        if label_shape[0] < self.label_shape[0]:
            msg = 'Attempts to reduce label count from %d to %d, not allowed.' \
                % (self.label_shape[0], label_shape[0])
            raise ValueError(msg)
        if label_shape[1] != self.provide_label[0][1][2]:
            msg = 'label_shape object width inconsistent: %d vs %d.' \
                % (self.provide_label[0][1][2], label_shape[1])
            raise ValueError(msg)

    def draw_next(self, color=None, thickness=2, mean=None, std=None,
                  waitKey=None, window_name='draw_next'):
        """Display next image with bounding boxes drawn.

        Parameters
        ----------
        color : tuple
            Bounding box color in RGB, use None for random color
        thickness : int
            Bounding box border thickness
        mean : True or numpy.ndarray
            Compensate for the mean to have better visual effect
        std : True or numpy.ndarray
            Revert standard deviations

        Returns
        -------
            image as numpy.ndarray
        """
        try:
            import cv2
        except ImportError as e:
            logging.warn('Unable to import cv2, skip drawing: %s', str(e))
            raise StopIteration
        count = 0
        try:
            while True:
                label, s = self.next_sample()
                data = self.imdecode(s)
                try:
                    self.check_valid_image([data])
                    label = self._parse_label(label)
                except RuntimeError as e:
                    logging.debug('Invalid image, skipping:  %s', str(e))
                    continue
                count += 1
                data, label = self.augmentation_transform(data, label)
                image = data.asnumpy()

                # revert color_normalize
                if std is True:
                    std = np.array([58.395, 57.12, 57.375])
                elif std is not None:
                    assert isinstance(std, np.ndarray) and std.shape[0] in [1, 3]
                if std is not None:
                    image *= std

                if mean is True:
                    mean = np.array([123.68, 116.28, 103.53])
                elif mean is not None:
                    assert isinstance(mean, np.ndarray) and mean.shape[0] in [1, 3]
                if mean is not None:
                    image += mean

                # swap RGB
                image[:, :, (0, 1, 2)] = image[:, :, (2, 1, 0)]
                if color:
                    color = color[::-1]
                image = image.astype(np.uint8)
                height, width, _ = image.shape
                for i in range(label.shape[0]):
                    x1 = int(label[i, 1] * width)
                    if x1 < 0:
                        continue
                    y1 = int(label[i, 2] * height)
                    x2 = int(label[i, 3] * width)
                    y2 = int(label[i, 4] * height)
                    bc = np.random.rand(3) * 255 if not color else color
                    cv2.rectangle(image, (x1, y1), (x2, y2), bc, thickness)
                if waitKey is not None:
                    cv2.imshow(window_name, image)
                    cv2.waitKey(waitKey)
                yield image
        except StopIteration:
            if not count:
                raise StopIteration


def synchronize_label_shape(train_iter, val_iter, verbose=False):
    """Synchronize label shape in train/validation iterators.

    Parameters
    ----------
    train_iter : ImageDetIter
        Training iterator
    val_iter : ImageDetIter
        Validation iterator
    verbose : bool
        Print verbose log if true

    Returns
    -------
    (ImageDetIter, ImageDetIter)
    """
    assert isinstance(train_iter, ImageDetIter) and isinstance(val_iter, ImageDetIter)
    train_label_shape = train_iter.label_shape
    val_label_shape = val_iter.label_shape
    assert train_label_shape[1] == val_label_shape[1], "object width mismatch."
    max_count = max(train_label_shape[0], val_label_shape[0])
    if max_count > train_label_shape[0]:
        train_iter.reshape(None, (max_count, train_label_shape[1]))
    if max_count > val_label_shape[0]:
        val_iter.reshape(None, (max_count, val_label_shape[1]))
    if verbose and max_count > min(train_label_shape[0], val_label_shape[0]):
        logging.info('Resized label_shape to (%d, %d).', max_count, train_label_shape[0])
    return (train_iter, val_iter)
