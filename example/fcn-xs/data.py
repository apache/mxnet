# pylint: skip-file
""" file iterator for pasval voc 2012"""
import mxnet as mx
import numpy as np
import sys, os
from mxnet.io import DataIter
from PIL import Image

class FileIter(DataIter):
    """FileIter object in fcn-xs example. Taking a file list file to get dataiter.
    in this example, we use the whole image training for fcn-xs, that is to say
    we do not need resize/crop the image to the same size, so the batch_size is
    set to 1 here
    Parameters
    ----------
    root_dir : string
        the root dir of image/label lie in
    flist_name : string
        the list file of iamge and label, every line owns the form:
        index \t image_data_path \t image_label_path
    cut_off_size : int
        if the maximal size of one image is larger than cut_off_size, then it will
        crop the image with the minimal size of that image
    data_name : string
        the data name used in symbol data(default data name)
    label_name : string
        the label name used in symbol softmax_label(default label name)
    """
    def __init__(self, root_dir, flist_name,
                 rgb_mean = (117, 117, 117),
                 cut_off_size = None,
                 data_name = "data",
                 label_name = "softmax_label"):
        super(FileIter, self).__init__()
        self.root_dir = root_dir
        self.flist_name = os.path.join(self.root_dir, flist_name)
        self.mean = np.array(rgb_mean)  # (R, G, B)
        self.cut_off_size = cut_off_size
        self.data_name = data_name
        self.label_name = label_name

        self.num_data = len(open(self.flist_name, 'r').readlines())
        self.f = open(self.flist_name, 'r')
        self.data, self.label = self._read()
        self.cursor = -1

    def _read(self):
        """get two list, each list contains two elements: name and nd.array value"""
        _, data_img_name, label_img_name = self.f.readline().strip('\n').split("\t")
        data = {}
        label = {}
        data[self.data_name], label[self.label_name] = self._read_img(data_img_name, label_img_name)
        return list(data.items()), list(label.items())

    def _read_img(self, img_name, label_name):
        img = Image.open(os.path.join(self.root_dir, img_name))
        label = Image.open(os.path.join(self.root_dir, label_name))
        assert img.size == label.size
        img = np.array(img, dtype=np.float32)  # (h, w, c)
        label = np.array(label)  # (h, w)
        if self.cut_off_size is not None:
            max_hw = max(img.shape[0], img.shape[1])
            min_hw = min(img.shape[0], img.shape[1])
            if min_hw > self.cut_off_size:
                rand_start_max = int(np.random.uniform(0, max_hw - self.cut_off_size - 1))
                rand_start_min = int(np.random.uniform(0, min_hw - self.cut_off_size - 1))
                if img.shape[0] == max_hw :
                    img = img[rand_start_max : rand_start_max + self.cut_off_size, rand_start_min : rand_start_min + self.cut_off_size]
                    label = label[rand_start_max : rand_start_max + self.cut_off_size, rand_start_min : rand_start_min + self.cut_off_size]
                else :
                    img = img[rand_start_min : rand_start_min + self.cut_off_size, rand_start_max : rand_start_max + self.cut_off_size]
                    label = label[rand_start_min : rand_start_min + self.cut_off_size, rand_start_max : rand_start_max + self.cut_off_size]
            elif max_hw > self.cut_off_size:
                rand_start = int(np.random.uniform(0, max_hw - min_hw - 1))
                if img.shape[0] == max_hw :
                    img = img[rand_start : rand_start + min_hw, :]
                    label = label[rand_start : rand_start + min_hw, :]
                else :
                    img = img[:, rand_start : rand_start + min_hw]
                    label = label[:, rand_start : rand_start + min_hw]
        reshaped_mean = self.mean.reshape(1, 1, 3)
        img = img - reshaped_mean
        img = np.swapaxes(img, 0, 2)
        img = np.swapaxes(img, 1, 2)  # (c, h, w)
        img = np.expand_dims(img, axis=0)  # (1, c, h, w)
        label = np.array(label)  # (h, w)
        label = np.expand_dims(label, axis=0)  # (1, h, w)
        return (img, label)

    @property
    def provide_data(self):
        """The name and shape of data provided by this iterator"""
        return [(k, tuple([1] + list(v.shape[1:]))) for k, v in self.data]

    @property
    def provide_label(self):
        """The name and shape of label provided by this iterator"""
        return [(k, tuple([1] + list(v.shape[1:]))) for k, v in self.label]

    def get_batch_size(self):
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
        """return one dict which contains "data" and "label" """
        if self.iter_next():
            self.data, self.label = self._read()
            return {self.data_name  :  self.data[0][1],
                    self.label_name :  self.label[0][1]}
        else:
            raise StopIteration
