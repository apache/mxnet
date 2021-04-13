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
# pylint: disable= arguments-differ, wildcard-import
"Contrib Vision DataLoaders."
import logging
import numpy as np

from ..... import ndarray as nd
from .....util import is_np_array
from ..... import numpy as _mx_np   # pylint: disable=reimported
from ....nn import HybridSequential, Sequential, HybridBlock, Block
from ....data.vision import transforms
from ....data import DataLoader
from .transforms import bbox

__all__ = ['create_image_augment', 'ImageDataLoader', 'ImageBboxDataLoader']

def create_image_augment(data_shape, resize=0, rand_crop=False, rand_resize=False, rand_mirror=False,
                         mean=None, std=None, brightness=0, contrast=0, saturation=0, hue=0,
                         pca_noise=0, rand_gray=0, inter_method=2, dtype='float32'):
    """Creates an augmenter block.

    Parameters
    ----------
    data_shape : tuple of int
        Shape for output data
    resize : int
        Resize shorter edge if larger than 0 at the begining
    rand_crop : bool
        Whether to enable random cropping other than center crop
    rand_resize : bool
        Whether to enable random sized cropping, require rand_crop to be enabled
    rand_gray : float
        [0, 1], probability to convert to grayscale for all channels, the number
        of channels will not be reduced to 1
    rand_mirror : bool
        Whether to apply horizontal flip to image with probability 0.5
    mean : np.ndarray or None
        Mean pixel values for [r, g, b]
    std : np.ndarray or None
        Standard deviations for [r, g, b]
    brightness : float
        Brightness jittering range (percent)
    contrast : float
        Contrast jittering range (percent)
    saturation : float
        Saturation jittering range (percent)
    hue : float
        Hue jittering range (percent)
    pca_noise : float
        Pca noise level (percent)
    inter_method : int, default=2(Area-based)
        Interpolation method for all resizing operations

        Possible values:
        0: Nearest Neighbors Interpolation.
        1: Bilinear interpolation.
        2: Bicubic interpolation over 4x4 pixel neighborhood.
        3: Area-based (resampling using pixel area relation). It may be a
        preferred method for image decimation, as it gives moire-free
        results. But when the image is zoomed, it is similar to the Nearest
        Neighbors method. (used by default).
        4: Lanczos interpolation over 8x8 pixel neighborhood.
        10: Random select from interpolation method metioned above.
        Note:
        When shrinking an image, it will generally look best with AREA-based
        interpolation, whereas, when enlarging an image, it will generally look best
        with Bicubic (slow) or Bilinear (faster but still looks OK).

    Examples
    --------
    >>> # An example of creating multiple augmenters
    >>> augs = mx.gluon.contrib.data.create_image_augment(data_shape=(3, 300, 300), rand_mirror=True,
    ...    mean=True, brightness=0.125, contrast=0.125, rand_gray=0.05,
    ...    saturation=0.125, pca_noise=0.05, inter_method=10)
    """
    if inter_method == 10:
        inter_method = np.random.randint(0, 5)
    augmenter = HybridSequential()
    if resize > 0:
        augmenter.add(transforms.image.Resize(resize, interpolation=inter_method))
    crop_size = (data_shape[2], data_shape[1])
    if rand_resize:
        assert rand_crop
        augmenter.add(transforms.image.RandomResizedCrop(crop_size, interpolation=inter_method))
    elif rand_crop:
        augmenter.add(transforms.image.RandomCrop(crop_size, interpolation=inter_method))
    else:
        augmenter.add(transforms.image.CenterCrop(crop_size, interpolation=inter_method))

    if rand_mirror:
        augmenter.add(transforms.image.RandomFlipLeftRight(0.5))

    augmenter.add(transforms.Cast())

    if brightness or contrast or saturation or hue:
        augmenter.add(transforms.image.RandomColorJitter(brightness, contrast, saturation, hue))

    if pca_noise > 0:
        augmenter.add(transforms.image.RandomLighting(pca_noise))

    if rand_gray > 0:
        augmenter.add(transforms.image.RandomGray(rand_gray))

    if mean is True:
        mean = [123.68, 116.28, 103.53]
    elif mean is not None:
        assert isinstance(mean, (tuple, list))

    if std is True:
        std = [58.395, 57.12, 57.375]
    elif std is not None:
        assert isinstance(std, (tuple, list))

    augmenter.add(transforms.image.ToTensor())

    if mean is not None or std is not None:
        augmenter.add(transforms.image.Normalize(mean, std))

    augmenter.add(transforms.Cast(dtype))

    return augmenter

class ImageDataLoader(object):
    """Image data loader with a large number of augmentation choices.
    This loader supports reading from both .rec files and raw image files.

    To load input images from .rec files, use `path_imgrec` parameter and to load from raw image
    files, use `path_imglist` and `path_root` parameters.

    To use data partition (for distributed training) or shuffling, specify `path_imgidx` parameter.

    Parameters
    ----------
    batch_size : int
        Number of examples per batch.
    data_shape : tuple
        Data shape in (channels, height, width) format.
        For now, only RGB image with 3 channels is supported.
    path_imgrec : str
        Path to image record file (.rec).
        Created with tools/im2rec.py or bin/im2rec.
    path_imglist : str
        Path to image list (.lst).
        Created with tools/im2rec.py or with custom script.
        Format: Tab separated record of index, one or more labels and relative_path_from_root.
    imglist: list
        A list of images with the label(s).
        Each item is a list [imagelabel: float or list of float, imgpath].
    path_root : str
        Root folder of image files.
        Whether to shuffle all images at the start of each iteration or not.
        Can be slow for HDD.
    part_index : int
        Partition index.
    num_parts : int
        Total number of partitions.
    dtype : str
        Label data type. Default: float32. Other options: int32, int64, float64
    last_batch : {'keep', 'discard', 'rollover'}
        How to handle the last batch if batch_size does not evenly divide
        `len(dataset)`.

        keep - A batch with less samples than previous batches is returned.
        discard - The last batch is discarded if its incomplete.
        rollover - The remaining samples are rolled over to the next epoch.
    kwargs : ...
        More arguments for creating augmenter. See mx.gluon.contrib.vision.dataloader.create_image_augment.
    """
    def __init__(self, batch_size, data_shape, path_imgrec=None, path_imglist=None, path_root='.',
                 part_index=0, num_parts=1, aug_list=None, imglist=None,
                 dtype='float32', shuffle=False, sampler=None,
                 last_batch=None, batch_sampler=None, batchify_fn=None,
                 num_workers=0, pin_memory=False, pin_device_id=0,
                 prefetch=None, thread_pool=False, timeout=120, try_nopython=None,
                 **kwargs):
        assert path_imgrec or path_imglist or (isinstance(imglist, list))
        assert dtype in ['int32', 'float32', 'int64', 'float64'], dtype + ' label not supported'
        logging.info('Using %s workers for decoding...', str(num_workers))
        logging.info('Set `num_workers` variable to a larger number to speed up loading'
                     ' (it requires shared memory to work and may occupy more memory).')
        class_name = self.__class__.__name__
        if path_imgrec:
            logging.info('%s: loading recordio %s...',
                         class_name, path_imgrec)
            from ....data.vision.datasets import ImageRecordDataset
            dataset = ImageRecordDataset(path_imgrec, flag=1)
        elif path_imglist:
            logging.info('%s: loading image list %s...', class_name, path_imglist)
            from ....data.vision.datasets import ImageListDataset
            dataset = ImageListDataset(path_root, path_imglist, flag=1)
        elif isinstance(imglist, list):
            logging.info('%s: loading image list...', class_name)
            from ....data.vision.datasets import ImageListDataset
            dataset = ImageListDataset(path_root, imglist, flag=1)
        else:
            raise ValueError('Either path_imgrec, path_imglist, or imglist must be provided')

        if num_parts > 1:
            dataset = dataset.shard(num_parts, part_index)

        if aug_list is None:
            # apply default transforms
            augmenter = create_image_augment(data_shape, **kwargs)
        elif isinstance(aug_list, list):
            if all([isinstance(a, HybridBlock) for a in aug_list]):
                augmenter = HybridSequential()
            else:
                augmenter = Sequential()
            for aug in aug_list:
                augmenter.add(aug)
        elif isinstance(aug_list, Block):
            augmenter = aug_list
        else:
            raise ValueError('aug_list must be a list of Blocks or Block')
        augmenter.hybridize()
        self._iter = DataLoader(dataset.transform_first(augmenter), batch_size=batch_size,
                                shuffle=shuffle, sampler=sampler, last_batch=last_batch,
                                batch_sampler=batch_sampler, batchify_fn=batchify_fn,
                                num_workers=num_workers, pin_memory=pin_memory,
                                pin_device_id=pin_device_id, prefetch=prefetch,
                                thread_pool=thread_pool, timeout=timeout, try_nopython=try_nopython)

    def __iter__(self):
        return iter(self._iter)

    def __len__(self):
        return len(self._iter)

def create_bbox_augment(data_shape, rand_crop=0, rand_pad=0, rand_gray=0,
                        rand_mirror=False, mean=None, std=None, brightness=0, contrast=0,
                        saturation=0, pca_noise=0, hue=0, inter_method=2,
                        max_aspect_ratio=2, area_range=(0.3, 3.0),
                        max_attempts=50, pad_val=(127, 127, 127), dtype='float32'):
    """Create augmenters for bbox/object detection.

    Parameters
    ----------
    data_shape : tuple of int
        Shape for output data
    rand_crop : float
        [0, 1], probability to apply random cropping
    rand_pad : float
        [0, 1], probability to apply random padding
    rand_gray : float
        [0, 1], probability to convert to grayscale for all channels
    rand_mirror : bool
        Whether to apply horizontal flip to image with probability 0.5
    mean : np.ndarray or None
        Mean pixel values for [r, g, b]
    std : np.ndarray or None
        Standard deviations for [r, g, b]
    brightness : float
        Brightness jittering range (percent)
    contrast : float
        Contrast jittering range (percent)
    saturation : float
        Saturation jittering range (percent)
    hue : float
        Hue jittering range (percent)
    pca_noise : float
        Pca noise level (percent)
    inter_method : int, default=2(Area-based)
        Interpolation method for all resizing operations

        Possible values:
        0: Nearest Neighbors Interpolation.
        1: Bilinear interpolation.
        2: Area-based (resampling using pixel area relation). It may be a
        preferred method for image decimation, as it gives moire-free
        results. But when the image is zoomed, it is similar to the Nearest
        Neighbors method. (used by default).
        3: Bicubic interpolation over 4x4 pixel neighborhood.
        4: Lanczos interpolation over 8x8 pixel neighborhood.
        10: Random select from interpolation method metioned above.
        Note:
        When shrinking an image, it will generally look best with AREA-based
        interpolation, whereas, when enlarging an image, it will generally look best
        with Bicubic (slow) or Bilinear (faster but still looks OK).
    max_aspect_ratio : float
        The cropped area of the image must have an aspect ratio = width / height
        within this range.
    area_range : tuple of floats
        The cropped area of the image must contain a fraction of the supplied
        image within in this range.
    max_attempts : int
        Number of attempts at generating a cropped/padded region of the image of the
        specified constraints. After max_attempts failures, return the original image.
    pad_val: float
        Pixel value to be filled when padding is enabled. pad_val will automatically
        be subtracted by mean and divided by std if applicable.

    Examples
    --------
    >>> # An example of creating multiple augmenters
    >>> augs = mx.gluon.contrib.data.create_bbox_augment(data_shape=(3, 300, 300), rand_crop=0.5,
    ...    rand_pad=0.5, rand_mirror=True, mean=True, brightness=0.125, contrast=0.125,
    ...    saturation=0.125, pca_noise=0.05, inter_method=10, min_object_covered=[0.3, 0.5, 0.9],
    ...    area_range=(0.3, 3.0))
    """
    if inter_method == 10:
        inter_method = np.random.randint(0, 5)
    augmenter = Sequential()
    if rand_crop > 0:
        augmenter.add(bbox.ImageBboxRandomCropWithConstraints(
            p=rand_crop, min_scale=area_range[0], max_scale=1.0,
            max_aspect_ratio=max_aspect_ratio, max_trial=max_attempts))

    if rand_mirror > 0:
        augmenter.add(bbox.ImageBboxRandomFlipLeftRight(0.5))

    if rand_pad > 0:
        augmenter.add(bbox.ImageBboxRandomExpand(
            p=rand_pad, max_ratio=area_range[1], fill=pad_val))

    # force resize
    augmenter.add(bbox.ImageBboxResize(data_shape[2], data_shape[1], interp=inter_method))

    if brightness or contrast or saturation or hue:
        augmenter.add(transforms.image.RandomColorJitter(
            brightness=brightness, contrast=contrast, saturation=saturation, hue=hue))

    if pca_noise > 0:
        augmenter.add(transforms.image.RandomLighting(pca_noise))

    if rand_gray > 0:
        augmenter.add(transforms.image.RandomGray(rand_gray))

    if mean is True:
        mean = [123.68, 116.28, 103.53]
    elif mean is not None:
        assert isinstance(mean, (tuple, list))

    if std is True:
        std = [58.395, 57.12, 57.375]
    elif std is not None:
        assert isinstance(std, (tuple, list))

    augmenter.add(transforms.image.ToTensor())
    if mean is not None or std is not None:
        augmenter.add(transforms.image.Normalize(mean, std))

    augmenter.add(transforms.Cast(dtype))

    return augmenter


class ImageBboxDataLoader(object):
    """Image iterator with a large number of augmentation choices for detection.

    Parameters
    ----------
    batch_size : int
        Number of examples per batch.
    data_shape : tuple
        Data shape in (channels, height, width) format.
        For now, only RGB image with 3 channels is supported.
    path_imgrec : str
        Path to image record file (.rec).
        Created with tools/im2rec.py or bin/im2rec.
    path_imglist : str
        Path to image list (.lst).
        Created with tools/im2rec.py or with custom script.
        Format: Tab separated record of index, one or more labels and relative_path_from_root.
    imglist: list
        A list of images with the label(s).
        Each item is a list [imagelabel: float or list of float, imgpath].
    path_root : str
        Root folder of image files.
    shuffle : bool
        Whether to shuffle all images at the start of each iteration or not.
        Can be slow for HDD.
    aug_list : list or None
        Augmenter list for generating distorted images
    part_index : int
        Partition index.
    num_parts : int
        Total number of partitions.
    last_batch : {'keep', 'discard', 'rollover'}
        How to handle the last batch if batch_size does not evenly divide
        `len(dataset)`.

        keep - A batch with less samples than previous batches is returned.
        discard - The last batch is discarded if its incomplete.
        rollover - The remaining samples are rolled over to the next epoch.
    kwargs : ...
        More arguments for creating augmenter. See mx.gluon.contrib.data.create_bbox_augment.
    """
    def __init__(self, batch_size, data_shape, path_imgrec=None, path_imglist=None, path_root='.',
                 part_index=0, num_parts=1, aug_list=None, imglist=None,
                 coord_normalized=True, dtype='float32', shuffle=False, sampler=None,
                 last_batch=None, batch_sampler=None, batchify_fn=None,
                 num_workers=0, pin_memory=False, pin_device_id=0,
                 prefetch=None, thread_pool=False, timeout=120, try_nopython=None,
                 **kwargs):
        assert path_imgrec or path_imglist or (isinstance(imglist, list))
        assert dtype in ['int32', 'float32', 'int64', 'float64'], dtype + ' label not supported'
        logging.info('Using %s workers for decoding...', str(num_workers))
        logging.info('Set `num_workers` variable to a larger number to speed up loading'
                     ' (it requires shared memory to work and may occupy more memory).')
        class_name = self.__class__.__name__
        if path_imgrec:
            logging.info('%s: loading recordio %s...',
                         class_name, path_imgrec)
            from ....data.vision.datasets import ImageRecordDataset
            dataset = ImageRecordDataset(path_imgrec, flag=1)
        elif path_imglist:
            logging.info('%s: loading image list %s...', class_name, path_imglist)
            from ....data.vision.datasets import ImageListDataset
            dataset = ImageListDataset(path_root, path_imglist, flag=1)
        elif isinstance(imglist, list):
            logging.info('%s: loading image list...', class_name)
            from ....data.vision.datasets import ImageListDataset
            dataset = ImageListDataset(path_root, imglist, flag=1)
        else:
            raise ValueError('Either path_imgrec, path_imglist, or imglist must be provided')

        if num_parts > 1:
            dataset = dataset.shard(num_parts, part_index)

        if aug_list is None:
            # apply default transforms
            augmenter = create_bbox_augment(data_shape, **kwargs)
        elif isinstance(aug_list, list):
            if all([isinstance(a, HybridBlock) for a in aug_list]):
                augmenter = HybridSequential()
            else:
                augmenter = Sequential()
            for aug in aug_list:
                augmenter.add(aug)
        elif isinstance(aug_list, Block):
            augmenter = aug_list
        else:
            raise ValueError('aug_list must be a list of Blocks')
        augmenter.hybridize()
        wrapper_aug = Sequential()
        wrapper_aug.add(BboxLabelTransform(coord_normalized))
        wrapper_aug.add(augmenter)

        if batchify_fn is None:
            from ....data.batchify import Stack, Pad, Group
            pad_batchify = Pad(val=-1)
            pad_batchify._warned = True
            batchify_fn = Group(Stack(), pad_batchify)  # stack image, pad bbox
        self._iter = DataLoader(dataset.transform(wrapper_aug), batch_size=batch_size,
                                shuffle=shuffle, sampler=sampler, last_batch=last_batch,
                                batch_sampler=batch_sampler, batchify_fn=batchify_fn,
                                num_workers=num_workers, pin_memory=pin_memory,
                                pin_device_id=pin_device_id, prefetch=prefetch,
                                thread_pool=thread_pool, timeout=timeout, try_nopython=try_nopython)

    def __iter__(self):
        return iter(self._iter)

    def __len__(self):
        return len(self._iter)

class BboxLabelTransform(Block):
    """Transform to convert 1-D bbox label to 2-D as in shape Nx5.

    Parameters
    ----------
    coord_normalized : bool
        Whether the coordinates(x0, y0, x1, y1) are normalized to (0, 1).

    """
    def __init__(self, coord_normalized=True):
        super(BboxLabelTransform, self).__init__()
        self._coord_normalized = coord_normalized

    def forward(self, img, label):
        """transform 1-D bbox label to Nx5 ndarray"""
        if self._coord_normalized:
            height = img.shape[0]
            width = img.shape[1]
        else:
            height = width = None
        if not isinstance(label, np.ndarray):
            label = label.asnumpy()
        label = label.flatten()
        header_len = int(label[0])  # label header
        label_width = int(label[1])  # the label width for each object, >= 5
        if label_width < 5:
            raise ValueError(
                "Label info for each object should >= 5, given {}".format(label_width))
        min_len = header_len + 5
        if len(label) < min_len:
            raise ValueError(
                "Expected label length >= {}, got {}".format(min_len, len(label)))
        if (len(label) - header_len) % label_width:
            raise ValueError(
                "Broken label of size {}, cannot reshape into (N, {}) "
                "if header length {} is excluded".format(len(label), label_width, header_len))
        bbox_label = label[header_len:].reshape(-1, label_width)
        # swap columns, requires [xmin-ymin-xmax-ymax-id-extra0-extra1-xxx]
        ids = bbox_label[:, 0].copy()
        bbox_label[:, :4] = bbox_label[:, 1:5]
        bbox_label[:, 4] = ids
        # restore to absolute coordinates
        if width is not None:
            bbox_label[:, (0, 2)] *= width
        if height is not None:
            bbox_label[:, (1, 3)] *= height
        array_fn = _mx_np.array if is_np_array() else nd.array
        return img, array_fn(bbox_label)
