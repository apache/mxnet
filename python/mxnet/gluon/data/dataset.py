# coding: utf-8
# pylint: disable=
"""Dataset container."""
import os

from ... import recordio, image

class Dataset(object):
    """Abstract dataset class. All datasets should have this interface.

    Subclasses need to override `__getitem__`, which returns the i-th
    element, and `__len__`, which returns the total number elements.

    .. note:: An mxnet or numpy array can be directly used as a dataset.
    """
    def __getitem__(self, idx):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError


class ArrayDataset(Dataset):
    """A dataset with a data array and a label array.

    The i-th sample is `(data[i], lable[i])`.

    Parameters
    ----------
    data : array-like object
        The data array. Can be mxnet or numpy array.
    label : array-like object
        The label array. Can be mxnet or numpy array.
    """
    def __init__(self, data, label):
        assert len(data) == len(label)
        self._data = data
        self._label = label

    def __getitem__(self, idx):
        return self._data[idx], self._label[idx]

    def __len__(self):
        return len(self._data)


class RecordFileDataset(Dataset):
    """A dataset wrapping over a RecordIO (.rec) file.

    Each sample is a string representing the raw content of an record.

    Parameters
    ----------
    filename : str
        Path to rec file.
    """
    def __init__(self, filename):
        idx_file = os.path.splitext(filename)[0] + '.idx'
        self._record = recordio.MXIndexedRecordIO(idx_file, filename, 'r')

    def __getitem__(self, idx):
        return self._record.read_idx(idx)

    def __len__(self):
        return len(self._record.keys)


class ImageRecordDataset(RecordFileDataset):
    """A dataset wrapping over a RecordIO file containing images.

    Each sample is an image and its corresponding label.

    Parameters
    ----------
    filename : str
        Path to rec file.
    flag : {0, 1}, default 1
        If 0, always convert images to greyscale.

        If 1, always convert images to colored (RGB).
    """
    def __init__(self, filename, flag=1):
        super(ImageRecordDataset, self).__init__(filename)
        self._flag = flag

    def __getitem__(self, idx):
        record = super(ImageRecordDataset, self).__getitem__(idx)
        header, img = recordio.unpack(record)
        return image.imdecode(img, self._flag), header.label
