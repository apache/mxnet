# pylint: skip-file
""" data iterator for pasval voc 2012"""
import mxnet as mx
import numpy as np
import sys
import os
from mxnet.io import DataIter
from skimage import io
from PIL import Image

class FileIter(DataIter):
    """FileIter object in mxnet. Taking NDArray or numpy array to get dataiter.
    Parameters
    ----------
    data_list or data, label: a list of, or two separate NDArray or numpy.ndarray
        list of NDArray for data. The last one is treated as label.
    batch_size: int
        Batch Size
    shuffle: bool
        Whether to shuffle the data
    data_pad_value: float, optional
        Padding value for data
    label_pad_value: float, optionl
        Padding value for label
    last_batch_handle: 'pad', 'discard' or 'roll_over'
        How to handle the last batch
    Note
    ----
    This iterator will pad, discard or roll over the last batch if
    the size of data does not match batch_size. Roll over is intended
    for training and can cause problems if used for prediction.
    """
    def __init__(self, root_dir, flist_name, data_name="data", label_name="softmax_label"):
        super(FileIter, self).__init__()
        self.root_dir = root_dir
        self.data_name = data_name
        self.label_name = label_name
        self.flist_name = os.path.join(self.root_dir, flist_name)
        self.num_data = len(open(self.flist_name, 'r').readlines())
        self.img_path = "img_path"
        self.f = open(self.flist_name, 'r')
        self.mean = np.array([123.68, 116.779, 103.939])  # (R, G, B)
        self.data, self.label, self.img_name = self._read(self.f)
        self.cursor = -1

    def _read(self, f):
        _, data_img_name, label_img_name = f.readline().strip('\n').split("\t")
        data = {}
        label = {}
        data[self.data_name] = self._read_img(data_img_name)
        label[self.label_name] = self._read_img(label_img_name, True)
        return list(data.items()), list(label.items()), data_img_name

    def _read_img(self, img_name, is_label_img=False):
        if not is_label_img:
            img = Image.open(os.path.join(self.root_dir, img_name))
            img = np.array(img, dtype=np.float32)  # (h, w, c)
            reshaped_mean = self.mean.reshape(1, 1, 3)
            img = img - reshaped_mean
            img = np.swapaxes(img, 0, 2)
            img = np.swapaxes(img, 1, 2)  # (c, h, w)
        else:
            img = Image.open(os.path.join(self.root_dir, img_name))
            img = np.array(img)  # (h, w)
            # img[img==255] = 0  # change the value of 255 to 0
        img = np.expand_dims(img, axis=0)  # (1, c, h, w) or (1, h, w)
        return img

    @property
    def provide_data(self):
        """The name and shape of data provided by this iterator"""
        return [(k, tuple([1] + list(v.shape[1:]))) for k, v in self.data]

    @property
    def provide_label(self):
        """The name and shape of label provided by this iterator"""
        return [(k, tuple([1] + list(v.shape[1:]))) for k, v in self.label]

    @property
    def batch_size(self):
        return 1

    def reset(self):
        self.cursor = -1
        self.f.close()
        self.f = open(self.flist_name, 'r')

    def iter_next(self):
        self.cursor += 1
        if(self.cursor < self.num_data-1):
            return True
        else:
            return False

    def next(self):
        if self.iter_next():
            self.data, self.label, self.img_name = self._read(self.f)
            return {self.data_name:self.getdata(),
                    self.label_name:self.getlabel(),
                    self.img_path:self.img_name}
        else:
            raise StopIteration

    def getdata(self):
        return self.data[0][1]

    def getlabel(self):
        return self.label[0][1]
