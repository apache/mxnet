# coding: utf-8
# pylint: disable=no-member, too-many-lines, redefined-builtin, protected-access, unused-import, invalid-name
# pylint: disable=too-many-arguments, too-many-locals, no-name-in-module, too-many-branches
"""Image IO API of mxnet."""
from __future__ import absolute_import

import os
import random
import numpy as np
from . import ndarray as nd
from . import _ndarray_internal as _internal
from ._ndarray_internal import _cvimresize as imresize
from ._ndarray_internal import _cvcopyMakeBorder as copyMakeBorder
from . import io
from . import recordio

def imdecode(buf, **kwargs):
    """Decode an image from string. Requires OpenCV to work.

    Parameters
    ----------
    buf : str/bytes, or numpy.ndarray
        Binary image data.
    flag : int
        0 for grayscale. 1 for colored.
    to_rgb : int
        0 for BGR format (OpenCV default). 1 for RGB format (MXNet default).
    out : NDArray
        Output buffer. Use None for automatic allocation.
    """
    if not isinstance(buf, nd.NDArray):
        buf = nd.array(np.frombuffer(buf, dtype=np.uint8), dtype=np.uint8)
    return _internal._cvimdecode(buf, **kwargs)

def scale_down(src_size, size):
    """Scale down crop size if it's bigger than image size"""
    w, h = size
    sw, sh = src_size
    if sh < h:
        w, h = float(w*sh)/h, sh
    if sw < w:
        w, h = sw, float(h*sw)/w
    return int(w), int(h)

def fixed_crop(src, x0, y0, w, h, size=None, interp=2):
    """Crop src at fixed location, and (optionally) resize it to size"""
    out = nd.crop(src, begin=(y0, x0, 0), end=(y0+h, x0+w, int(src.shape[2])))
    if size is not None and (w, h) != size:
        out = imresize(out, *size, interp=interp)
    return out

def random_crop(src, size, interp=2):
    """Randomly crop src with size. Upsample result if src is smaller than size"""
    h, w, _ = src.shape
    new_w, new_h = scale_down((w, h), size)

    x0 = random.randint(0, w - new_w)
    y0 = random.randint(0, h - new_h)

    out = fixed_crop(src, x0, y0, new_w, new_h, size, interp)
    return out, (x0, y0, new_w, new_h)

def center_crop(src, size, interp=2):
    """Randomly crop src with size. Upsample result if src is smaller than size"""
    h, w, _ = src.shape
    new_w, new_h = scale_down((w, h), size)

    x0 = (w - new_w)/2
    y0 = (h - new_h)/2

    out = fixed_crop(src, x0, y0, new_w, new_h, size, interp)
    return out, (x0, y0, new_w, new_h)

def color_normalize(src, mean, std):
    """Normalize src with mean and std"""
    src = src - mean
    if std is not None:
        src = src / std
    return src

def random_size_crop(src, size, min_area=0.08, ratio=(3.0/4.0, 4.0/3.0), interp=2):
    """Randomly crop src with size. Randomize area and aspect ratio"""
    h, w, _ = src.shape
    area = w*h
    for _ in range(10):
        new_area = random.uniform(min_area, 1.0) * area
        new_ratio = random.uniform(*ratio)
        new_w = int(np.sqrt(new_area*new_ratio))
        new_h = int(np.sqrt(new_area/new_ratio))

        if random.random() < 0.5:
            new_w, new_h = new_h, new_w

        if new_w > w or new_h > h:
            continue

        x0 = random.randint(0, w - new_w)
        y0 = random.randint(0, h - new_h)

        out = fixed_crop(src, x0, y0, new_w, new_h, size, interp)
        return out, (x0, y0, new_w, new_h)

    return random_crop(src, size)

def RandomCropAug(size, interp=2):
    """Make random crop augumenter"""
    def aug(src):
        """Augumenter body"""
        return random_crop(src, size, interp)[0]
    return aug

def RandomSizedCropAug(size, min_area=0.08, ratio=(3.0/4.0, 4.0/3.0), interp=2):
    """Make random crop with random resizing and random aspect ratio jitter augumenter"""
    def aug(src):
        """Augumenter body"""
        return random_size_crop(src, size, min_area, ratio, interp)[0]
    return aug

def CenterCropAug(size, interp=2):
    """Make center crop augmenter"""
    def aug(src):
        """Augumenter body"""
        return center_crop(src, size, interp)[0]
    return aug

def RandomOrderAug(ts):
    """Apply list of augmenters in random order"""
    def aug(src):
        """Augumenter body"""
        random.shuffle(ts)
        for i in ts:
            src = i(src)
        return src
    return aug

def ColorJitterAug(brightness, contrast, saturation):
    """Apply random brightness, contrast and saturation jitter in random order"""
    ts = []
    coef = nd.array([[[0.299, 0.587, 0.114]]])
    if brightness > 0:
        def baug(src):
            """Augumenter body"""
            alpha = 1.0 + random.uniform(-brightness, brightness)
            src *= alpha
            return src
        ts.append(baug)

    if contrast > 0:
        def caug(src):
            """Augumenter body"""
            alpha = 1.0 + random.uniform(-contrast, contrast)
            gray = src*coef
            gray = (3.0*(1.0-alpha)/gray.size)*nd.sum(gray)
            src *= alpha
            src = src + gray
            return src
        ts.append(caug)

    if saturation > 0:
        def saug(src):
            """Augumenter body"""
            alpha = 1.0 + random.uniform(-saturation, saturation)
            gray = src*coef
            gray = nd.sum(gray, axis=2, keepdims=True)
            gray *= (1.0-alpha)
            src *= alpha
            src = src + gray
            return src
        ts.append(saug)
    return RandomOrderAug(ts)

def LightingAug(alphastd, eigval, eigvec):
    """Add PCA based noise"""
    def aug(src):
        """Augumenter body"""
        alpha = np.random.normal(0, alphastd, size=(3,))
        rgb = np.dot(eigvec*alpha, eigval)
        src = src + nd.array(rgb)
        return src
    return aug

def ColorNormalizeAug(mean, std):
    """Mean and std normalization"""
    mean = nd.array(mean)
    std = nd.array(std)
    def aug(src):
        """Augumenter body"""
        return color_normalize(src, mean, std)
    return aug

def HorizontalFlipAug(p):
    """Random horizontal flipping"""
    def aug(src):
        """Augumenter body"""
        if random.random() < p:
            src = nd.flip(src, axis=1)
        return src
    return aug

def CastAug():
    """Cast to float32"""
    def aug(src):
        """Augumenter body"""
        src = src.astype(np.float32)
        return src
    return aug

def CreateAugmenter(data_shape, rand_crop=False, rand_resize=False, rand_mirror=False,
                    mean=None, std=None, brightness=0, contrast=0, saturation=0,
                    pca_noise=0, inter_method=2):
    """Create augumenter list"""
    auglist = []
    crop_size = (data_shape[2], data_shape[1])
    if rand_resize:
        assert rand_crop
        auglist.append(RandomSizedCropAug(crop_size, inter_method))
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
    if std is True:
        std = np.array([58.395, 57.12, 57.375])
    if mean:
        auglist.append(ColorNormalizeAug(mean, std))

    return auglist


class ImageIter(io.DataIter):
    """Image data iterator with a large number of augumentation choices.
    Supports reading from both .rec files and raw image files with image list.

    To load from .rec files, please specify path_imgrec. Also specify path_imgidx
    to use data partition (for distributed training) or shuffling.

    To load from raw image files, specify path_imglist and path_root.

    Parameters
    ----------
    batch_size : int
        Number of examples per batch
    data_shape : tuple
        Data shape in (channels, height, width).
        For now, only RGB image with 3 channels is supported.
    label_width : int
        dimension of label
    path_imgrec : str
        path to image record file (.rec).
        Created with tools/im2rec.py or bin/im2rec
    path_imglist : str
        path to image list (.lst)
        Created with tools/im2rec.py or with custom script.
        Format: index\t[one or more label separated by \t]\trelative_path_from_root
    path_root : str
        Root folder of image files
    path_imgidx : str
        Path to image index file. Needed for partition and shuffling when using .rec source.
    shuffle : bool
        Whether to shuffle all images at the start of each iteration.
        Can be slow for HDD.
    part_index : int
        Partition index
    num_parts : int
        Total number of partitions.
    kwargs : ...
        More arguments for creating augumenter. See mx.image.CreateAugmenter
    """
    def __init__(self, batch_size, data_shape, label_width=1,
                 path_imgrec=None, path_imglist=None, path_root=None, path_imgidx=None,
                 shuffle=False, part_index=0, num_parts=1, **kwargs):
        super(ImageIter, self).__init__()
        assert path_imgrec or path_imglist
        if path_imgrec:
            if path_imgidx:
                self.imgrec = recordio.MXIndexedRecordIO(path_imgidx, path_imgrec, 'r')
                self.imgidx = self.imgrec.idx
            else:
                self.imgrec = recordio.MXRecordIO(path_imgrec, 'r')
        else:
            self.imgrec = None
        if path_imglist:
            with open(path_imglist) as fin:
                imglist = {}
                while True:
                    line = fin.readline()
                    if not line:
                        break
                    line = line.strip().split('\t')
                    label = nd.array([float(i) for i in line[1:-1]])
                    imglist[int(line[0])] = (label, line[-1])
                self.imglist = imglist
        else:
            self.imglist = None
        self.path_root = path_root

        assert len(data_shape) == 3 and data_shape[0] == 3
        self.provide_data = [('data', (batch_size,) + data_shape)]
        self.provide_label = [('softmax_label', (batch_size, label_width))]
        self.batch_size = batch_size
        self.data_shape = data_shape
        self.label_width = label_width

        self.shuffle = shuffle
        if shuffle or num_parts > 1:
            if self.imgrec is None:
                self.seq = self.imglist.keys()
            else:
                assert self.imgidx is not None
                self.seq = self.imgidx.keys()
        else:
            self.seq = None

        if num_parts > 1:
            assert part_index < num_parts
            N = len(self.seq)
            C = N/num_parts
            self.seq = self.seq[part_index*C:(part_index+1)*C]

        self.auglist = CreateAugmenter(data_shape, **kwargs)
        self.cur = 0
        self.reset()

    def reset(self):
        if self.shuffle:
            random.shuffle(self.seq)
        if self.imgrec is not None:
            self.imgrec.reset()
        self.cur = 0

    def next_sample(self):
        """helper function for reading in next sample"""
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
                label, fname = self.imglist[self.seq[self.cur]]
                if self.imgrec is None:
                    with open(os.path.join(self.path_root, fname), 'rb') as fin:
                        img = fin.read()
                return label, img
        else:
            s = self.imgrec.read()
            header, img = recordio.unpack(s)
            return header.label, img

    def next(self):
        batch_size = self.batch_size
        c, h, w = self.data_shape
        batch_data = nd.zeros((batch_size, h, w, c))
        batch_label = nd.zeros((batch_size, self.label_width))
        i = 0
        try:
            for i in range(batch_size):
                label, data = self.next_sample()
                data = imdecode(data)
                for aug in self.auglist:
                    data = aug(data)
                batch_data[i][:] = data
                batch_label[i][:] = label
        except StopIteration:
            if not i:
                raise StopIteration

        batch_data = nd.transpose(batch_data, axes=(0, 3, 1, 2))
        return io.DataBatch(batch_data, batch_label, batch_size-1-i)
