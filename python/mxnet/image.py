# pylint: disable=no-member, too-many-lines, redefined-builtin, protected-access, unused-import, invalid-name
# pylint: disable=too-many-arguments, too-many-locals, no-name-in-module, too-many-branches, too-many-statements
"""Read individual image files and perform augmentations."""

from __future__ import absolute_import, print_function

import os
import random
import logging
import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

from .base import numeric_types
from . import ndarray as nd
from . import _ndarray_internal as _internal
from ._ndarray_internal import _cvimresize as imresize
from ._ndarray_internal import _cvcopyMakeBorder as copyMakeBorder
from . import io
from . import recordio


def imdecode(buf, **kwargs):
    """Decode an image to an NDArray.

    Note: `imdecode` uses OpenCV (not the CV2 Python library).
    MXNet must have been built with OpenCV for `imdecode` to work.

    Parameters
    ----------
    buf : str/bytes or numpy.ndarray
        Binary image data as string or numpy ndarray.
    flag : int, optional, default=1
        1 for three channel color output. 0 for grayscale output.
    to_rgb : int, optional, default=1
        1 for RGB formatted output (MXNet default). 0 for BGR formatted output (OpenCV default).
    out : NDArray, optional
        Output buffer. Use `None` for automatic allocation.

    Returns
    -------
    NDArray
        An `NDArray` containing the image.

    Example
    -------
    >>> with open("flower.jpg", 'rb') as fp:
    ...     str_image = fp.read()
    ...
    >>> image = mx.img.imdecode(str_image)
    >>> image
    <NDArray 224x224x3 @cpu(0)>

    Set `flag` parameter to 0 to get grayscale output

    >>> with open("flower.jpg", 'rb') as fp:
    ...     str_image = fp.read()
    ...
    >>> image = mx.img.imdecode(str_image, flag=0)
    >>> image
    <NDArray 224x224x1 @cpu(0)>

    Set `to_rgb` parameter to 0 to get output in OpenCV format (BGR)

    >>> with open("flower.jpg", 'rb') as fp:
    ...     str_image = fp.read()
    ...
    >>> image = mx.img.imdecode(str_image, to_rgb=0)
    >>> image
    <NDArray 224x224x3 @cpu(0)>
    """
    if not isinstance(buf, nd.NDArray):
        buf = nd.array(np.frombuffer(buf, dtype=np.uint8), dtype=np.uint8)
    return _internal._cvimdecode(buf, **kwargs)


def scale_down(src_size, size):
    """Scales down crop size if it's larger than image size.

    If width/height of the crop is larger than the width/height of the image,
    sets the width/height to the width/height of the image.

    Parameters
    ----------
    src_size : tuple of int
        Size of the image in (width, height) format.
    size : tuple of int
        Size of the crop in (width, height) format.

    Returns
    -------
    tuple of int
        A tuple containing the scaled crop size in (width, height) format.

    Example
    --------
    >>> src_size = (640,480)
    >>> size = (720,120)
    >>> new_size = mx.img.scale_down(src_size, size)
    >>> new_size
    (640,106)
    """
    w, h = size
    sw, sh = src_size
    if sh < h:
        w, h = float(w * sh) / h, sh
    if sw < w:
        w, h = sw, float(h * sw) / w
    return int(w), int(h)


def resize_short(src, size, interp=2):
    """Resizes shorter edge to size.

    Note: `resize_short` uses OpenCV (not the CV2 Python library).
    MXNet must have been built with OpenCV for `resize_short` to work.

    Resizes the original image by setting the shorter edge to size
    and setting the longer edge accordingly.
    Resizing function is called from OpenCV.

    Parameters
    ----------
    src : NDArray
        The original image.
    size : int
        The length to be set for the shorter edge.
    interp : int, optional, default=2
        Interpolation method used for resizing the image.
        Default method is bicubic interpolation.
        More details can be found in the documentation of OpenCV, please refer to
        http://docs.opencv.org/master/da/d54/group__imgproc__transform.html.

    Returns
    -------
    NDArray
        An 'NDArray' containing the resized image.

    Example
    -------
    >>> with open("flower.jpeg", 'rb') as fp:
    ...     str_image = fp.read()
    ...
    >>> image = mx.img.imdecode(str_image)
    >>> image
    <NDArray 2321x3482x3 @cpu(0)>
    >>> size = 640
    >>> new_image = mx.img.resize_short(image, size)
    >>> new_image
    <NDArray 2321x3482x3 @cpu(0)>
    """
    h, w, _ = src.shape
    if h > w:
        new_h, new_w = size * h / w, size
    else:
        new_h, new_w = size, size * w / h
    return imresize(src, new_w, new_h, interp=interp)


def fixed_crop(src, x0, y0, w, h, size=None, interp=2):
    """Crop src at fixed location, and (optionally) resize it to size."""
    out = nd.crop(src, begin=(y0, x0, 0), end=(y0 + h, x0 + w, int(src.shape[2])))
    if size is not None and (w, h) != size:
        out = imresize(out, *size, interp=interp)
    return out


def random_crop(src, size, interp=2):
    """Randomly crop `src` with `size` (width, height).
    Upsample result if `src` is smaller than `size`.

    Parameters
    ----------
    src: Source image `NDArray`
    size: Size of the crop formatted as (width, height). If the `size` is larger
           than the image, then the source image is upsampled to `size` and returned.
    interp: Interpolation method to be used in case the size is larger (default: bicubic).
            Uses OpenCV convention for the parameters. Nearest - 0, Bilinear - 1, Bicubic - 2,
            Area - 3. See OpenCV imresize function for more details.
    Returns
    -------
    NDArray
        An `NDArray` containing the cropped image.
    Tuple
        A tuple (x, y, width, height) where (x, y) is top-left position of the crop in the
        original image and (width, height) are the dimensions of the cropped image.

    Example
    -------
    >>> im = mx.nd.array(cv2.imread("flower.jpg"))
    >>> cropped_im, rect  = mx.image.random_crop(im, (100, 100))
    >>> print cropped_im
    <NDArray 100x100x1 @cpu(0)>
    >>> print rect
    (20, 21, 100, 100)
    """

    h, w, _ = src.shape
    new_w, new_h = scale_down((w, h), size)

    x0 = random.randint(0, w - new_w)
    y0 = random.randint(0, h - new_h)

    out = fixed_crop(src, x0, y0, new_w, new_h, size, interp)
    return out, (x0, y0, new_w, new_h)


def center_crop(src, size, interp=2):
    """Crops the image `src` to the given `size` by trimming on all four
    sides and preserving the center of the image. Upsamples if `src` is smaller
    than `size`.

    .. note:: This requires MXNet to be compiled with USE_OPENCV.

    Parameters
    ----------
    src : NDArray
        Binary source image data.
    size : list or tuple of int
        The desired output image size.
    interp : interpolation, optional, default=Area-based
        The type of interpolation that is done to the image.

         Possible values:

        0: Nearest Neighbors Interpolation.

        1: Bilinear interpolation.

        2: Area-based (resampling using pixel area relation). It may be a
        preferred method for image decimation, as it gives moire-free
        results. But when the image is zoomed, it is similar to the Nearest
        Neighbors method. (used by default).

        3: Bicubic interpolation over 4x4 pixel neighborhood.

        4: Lanczos interpolation over 8x8 pixel neighborhood.

        When shrinking an image, it will generally look best with AREA-based
        interpolation, whereas, when enlarging an image, it will generally look best
        with Bicubic (slow) or Bilinear (faster but still looks OK).

    Returns
    -------
    NDArray
        The cropped image.
    Tuple
        (x, y, width, height) where x, y are the positions of the crop in the
        original image and width, height the dimensions of the crop.

    Example
    -------
    >>> with open("flower.jpg", 'rb') as fp:
    ...     str_image = fp.read()
    ...
    >>> image = mx.image.imdecode(str_image)
    >>> image
    <NDArray 2321x3482x3 @cpu(0)>
    >>> cropped_image, (x, y, width, height) = mx.image.center_crop(image, (1000, 500))
    >>> cropped_image
    <NDArray 500x1000x3 @cpu(0)>
    >>> x, y, width, height
    (1241, 910, 1000, 500)
    """

    h, w, _ = src.shape
    new_w, new_h = scale_down((w, h), size)

    x0 = int((w - new_w) / 2)
    y0 = int((h - new_h) / 2)

    out = fixed_crop(src, x0, y0, new_w, new_h, size, interp)
    return out, (x0, y0, new_w, new_h)


def color_normalize(src, mean, std=None):
    """Normalize src with mean and std."""
    src -= mean
    if std is not None:
        src /= std
    return src


def random_size_crop(src, size, min_area, ratio, interp=2):
    """Randomly crop src with size. Randomize area and aspect ratio."""
    h, w, _ = src.shape
    new_ratio = random.uniform(*ratio)
    if new_ratio * h > w:
        max_area = w * int(w / new_ratio)
    else:
        max_area = h * int(h * new_ratio)

    min_area *= h * w
    if max_area < min_area:
        return random_crop(src, size, interp)
    new_area = random.uniform(min_area, max_area)
    new_w = int(np.sqrt(new_area * new_ratio))
    new_h = int(np.sqrt(new_area / new_ratio))

    assert new_w <= w and new_h <= h
    x0 = random.randint(0, w - new_w)
    y0 = random.randint(0, h - new_h)

    out = fixed_crop(src, x0, y0, new_w, new_h, size, interp)
    return out, (x0, y0, new_w, new_h)


def ResizeAug(size, interp=2):
    """Make resize shorter edge to size augmenter."""

    def aug(src):
        """Augmenter body"""
        return [resize_short(src, size, interp)]

    return aug


def RandomCropAug(size, interp=2):
    """Make random crop augmenter"""

    def aug(src):
        """Augmenter body"""
        return [random_crop(src, size, interp)[0]]

    return aug


def RandomSizedCropAug(size, min_area, ratio, interp=2):
    """Make random crop with random resizing and random aspect ratio jitter augmenter."""

    def aug(src):
        """Augmenter body"""
        return [random_size_crop(src, size, min_area, ratio, interp)[0]]

    return aug


def CenterCropAug(size, interp=2):
    """Make center crop augmenter."""

    def aug(src):
        """Augmenter body"""
        return [center_crop(src, size, interp)[0]]

    return aug


def RandomOrderAug(ts):
    """Apply list of augmenters in random order"""

    def aug(src):
        """Augmenter body"""
        src = [src]
        random.shuffle(ts)
        for t in ts:
            src = [j for i in src for j in t(i)]
        return src

    return aug


def ColorJitterAug(brightness, contrast, saturation):
    """Apply random brightness, contrast and saturation jitter in random order."""
    ts = []
    coef = nd.array([[[0.299, 0.587, 0.114]]])
    if brightness > 0:
        def baug(src):
            """Augmenter body"""
            alpha = 1.0 + random.uniform(-brightness, brightness)
            src *= alpha
            return [src]

        ts.append(baug)

    if contrast > 0:
        def caug(src):
            """Augmenter body"""
            alpha = 1.0 + random.uniform(-contrast, contrast)
            gray = src * coef
            gray = (3.0 * (1.0 - alpha) / gray.size) * nd.sum(gray)
            src *= alpha
            src += gray
            return [src]

        ts.append(caug)

    if saturation > 0:
        def saug(src):
            """Augmenter body"""
            alpha = 1.0 + random.uniform(-saturation, saturation)
            gray = src * coef
            gray = nd.sum(gray, axis=2, keepdims=True)
            gray *= (1.0 - alpha)
            src *= alpha
            src += gray
            return [src]

        ts.append(saug)
    return RandomOrderAug(ts)


def LightingAug(alphastd, eigval, eigvec):
    """Add PCA based noise."""

    def aug(src):
        """Augmenter body"""
        alpha = np.random.normal(0, alphastd, size=(3,))
        rgb = np.dot(eigvec * alpha, eigval)
        src += nd.array(rgb)
        return [src]

    return aug


def ColorNormalizeAug(mean, std):
    """Mean and std normalization."""
    mean = nd.array(mean)
    std = nd.array(std)

    def aug(src):
        """Augmenter body"""
        return [color_normalize(src, mean, std)]

    return aug


def HorizontalFlipAug(p):
    """Random horizontal flipping."""

    def aug(src):
        """Augmenter body"""
        if random.random() < p:
            src = nd.flip(src, axis=1)
        return [src]

    return aug


def CastAug():
    """Cast to float32"""

    def aug(src):
        """Augmenter body"""
        src = src.astype(np.float32)
        return [src]

    return aug


def CreateAugmenter(data_shape, resize=0, rand_crop=False, rand_resize=False, rand_mirror=False,
                    mean=None, std=None, brightness=0, contrast=0, saturation=0,
                    pca_noise=0, inter_method=2):
    """Creates an augmenter list."""
    auglist = []

    if resize > 0:
        auglist.append(ResizeAug(resize, inter_method))

    crop_size = (data_shape[2], data_shape[1])
    if rand_resize:
        assert rand_crop
        auglist.append(RandomSizedCropAug(crop_size, 0.3, (3.0 / 4.0, 4.0 / 3.0), inter_method))
    elif rand_crop:
        auglist.append(RandomCropAug(crop_size, inter_method))
    else:
        auglist.append(CenterCropAug(crop_size, inter_method))

    if rand_mirror:
        auglist.append(HorizontalFlipAug(0.5))

    auglist.append(CastAug())

    if brightness or contrast or saturation:
        auglist.append(ColorJitterAug(brightness, contrast, saturation))

    if pca_noise > 0:
        eigval = np.array([55.46, 4.794, 1.148])
        eigvec = np.array([[-0.5675, 0.7192, 0.4009],
                           [-0.5808, -0.0045, -0.8140],
                           [-0.5836, -0.6948, 0.4203]])
        auglist.append(LightingAug(pca_noise, eigval, eigvec))

    if mean is True:
        mean = np.array([123.68, 116.28, 103.53])
    elif mean is not None:
        assert isinstance(mean, np.ndarray) and mean.shape[0] in [1, 3]

    if std is True:
        std = np.array([58.395, 57.12, 57.375])
    elif std is not None:
        assert isinstance(std, np.ndarray) and std.shape[0] in [1, 3]

    if mean is not None and std is not None:
        auglist.append(ColorNormalizeAug(mean, std))

    return auglist


class ImageIter(io.DataIter):
    """Image data iterator with a large number of augmentation choices.
    This iterator supports reading from both .rec files and raw image files.

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
    label_width : int, optional
        Number of labels per example. The default label width is 1.
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
    path_imgidx : str
        Path to image index file. Needed for partition and shuffling when using .rec source.
    shuffle : bool
        Whether to shuffle all images at the start of each iteration or not.
        Can be slow for HDD.
    part_index : int
        Partition index.
    num_parts : int
        Total number of partitions.
    data_name : str
        Data name for provided symbols.
    label_name : str
        Label name for provided symbols.
    kwargs : ...
        More arguments for creating augmenter. See mx.image.CreateAugmenter.
    """

    def __init__(self, batch_size, data_shape, label_width=1,
                 path_imgrec=None, path_imglist=None, path_root=None, path_imgidx=None,
                 shuffle=False, part_index=0, num_parts=1, aug_list=None, imglist=None,
                 data_name='data', label_name='softmax_label', **kwargs):
        super(ImageIter, self).__init__()
        assert path_imgrec or path_imglist or (isinstance(imglist, list))
        if path_imgrec:
            print('loading recordio...')
            if path_imgidx:
                self.imgrec = recordio.MXIndexedRecordIO(path_imgidx, path_imgrec, 'r')  # pylint: disable=redefined-variable-type
                self.imgidx = list(self.imgrec.keys)
            else:
                self.imgrec = recordio.MXRecordIO(path_imgrec, 'r')  # pylint: disable=redefined-variable-type
                self.imgidx = None
        else:
            self.imgrec = None

        if path_imglist:
            print('loading image list...')
            with open(path_imglist) as fin:
                imglist = {}
                imgkeys = []
                for line in iter(fin.readline, ''):
                    line = line.strip().split('\t')
                    label = nd.array([float(i) for i in line[1:-1]])
                    key = int(line[0])
                    imglist[key] = (label, line[-1])
                    imgkeys.append(key)
                self.imglist = imglist
        elif isinstance(imglist, list):
            print('loading image list...')
            result = {}
            imgkeys = []
            index = 1
            for img in imglist:
                key = str(index)  # pylint: disable=redefined-variable-type
                index += 1
                if isinstance(img[0], numeric_types):
                    label = nd.array([img[0]])
                else:
                    label = nd.array(img[0])
                result[key] = (label, img[1])
                imgkeys.append(str(key))
            self.imglist = result
        else:
            self.imglist = None
        self.path_root = path_root

        self.check_data_shape(data_shape)
        self.provide_data = [(data_name, (batch_size,) + data_shape)]
        if label_width > 1:
            self.provide_label = [(label_name, (batch_size, label_width))]
        else:
            self.provide_label = [(label_name, (batch_size,))]
        self.batch_size = batch_size
        self.data_shape = data_shape
        self.label_width = label_width

        self.shuffle = shuffle
        if self.imgrec is None:
            self.seq = imgkeys
        elif shuffle or num_parts > 1:
            assert self.imgidx is not None
            self.seq = self.imgidx
        else:
            self.seq = None

        if num_parts > 1:
            assert part_index < num_parts
            N = len(self.seq)
            C = N / num_parts
            self.seq = self.seq[part_index * C:(part_index + 1) * C]
        if aug_list is None:
            self.auglist = CreateAugmenter(data_shape, **kwargs)
        else:
            self.auglist = aug_list
        self.cur = 0
        self.reset()

    def reset(self):
        """Resets the iterator to the beginning of the data."""
        if self.shuffle:
            random.shuffle(self.seq)
        if self.imgrec is not None:
            self.imgrec.reset()
        self.cur = 0

    def next_sample(self):
        """Helper function for reading in next sample."""
        if self.seq is not None:
            if self.cur >= len(self.seq):
                raise StopIteration
            idx = self.seq[self.cur]
            self.cur += 1
            if self.imgrec is not None:
                s = self.imgrec.read_idx(idx)
                header, img = recordio.unpack(s)
                if self.imglist is None:
                    return header.label, img
                else:
                    return self.imglist[idx][0], img
            else:
                label, fname = self.imglist[idx]
                return label, self.read_image(fname)
        else:
            s = self.imgrec.read()
            if s is None:
                raise StopIteration
            header, img = recordio.unpack(s)
            return header.label, img

    def next(self):
        """Returns the next batch of data."""
        batch_size = self.batch_size
        c, h, w = self.data_shape
        batch_data = nd.empty((batch_size, c, h, w))
        batch_label = nd.empty(self.provide_label[0][1])
        i = 0
        try:
            while i < batch_size:
                label, s = self.next_sample()
                data = [self.imdecode(s)]
                try:
                    self.check_valid_image(data)
                except RuntimeError as e:
                    logging.debug('Invalid image, skipping:  %s', str(e))
                    continue
                data = self.augmentation_transform(data)
                for datum in data:
                    assert i < batch_size, 'Batch size must be multiples of augmenter output length'
                    batch_data[i][:] = self.postprocess_data(datum)
                    batch_label[i][:] = label
                    i += 1
        except StopIteration:
            if not i:
                raise StopIteration

        return io.DataBatch([batch_data], [batch_label], batch_size - i)

    def check_data_shape(self, data_shape):
        """Checks if the input data shape is valid"""
        if not len(data_shape) == 3:
            raise ValueError('data_shape should have length 3, with dimensions CxHxW')
        if not data_shape[0] == 3:
            raise ValueError('This iterator expects inputs to have 3 channels.')

    def check_valid_image(self, data):
        """Checks if the input data is valid"""
        if len(data[0].shape) == 0:
            raise RuntimeError('Data shape is wrong')

    def imdecode(self, s):
        """Decodes a string or byte string to an NDArray.
        See mx.img.imdecode for more details."""
        return imdecode(s)

    def read_image(self, fname):
        """Reads an input image `fname` and returns the decoded raw bytes.

        Example usage:
        ----------
        >>> dataIter.read_image('Face.jpg') # returns decoded raw bytes.
        """
        with open(os.path.join(self.path_root, fname), 'rb') as fin:
            img = fin.read()
        return img

    def augmentation_transform(self, data):
        """Transforms input data with specified augmentation."""
        for aug in self.auglist:
            data = [ret for src in data for ret in aug(src)]
        return data

    def postprocess_data(self, datum):
        """Final postprocessing step before image is loaded into the batch."""
        return nd.transpose(datum, axes=(2, 0, 1))
