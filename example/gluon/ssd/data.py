# pylint: skip-file
""" data iterator for mnist """
import os
import random
import sys
import mxnet as mx
import math


class DetMNISTIter(mx.io.DataIter):
    """Synthetic detection dataset from MNIST by randomly place digits and
    record its position as label.

    Parameters:
    ----------
    image : str
        MNIST image file path.
    label : str
        MNIST label file path.
    batch_size : int
        Batch size.
    data_shape : int
        The new data shape as output size.
    max_ratio : float
        Maximum aspect ratio of generated digits.
    min_size : float
        Minimum ratio of digits in the new image.
    shuffle : bool
        Whether to randomize the image order.
    """
    def __init__(self, image, label, batch_size, data_shape=64, max_ratio=1.5,
                 min_size=0.3, shuffle=False):
        super(DetMNISTIter, self).__init__()
        self.iter = mx.io.MNISTIter(
            image=image,
            label=label,
            batch_size=batch_size,
            shuffle=shuffle,
            flat=False)
        self.data_shape = data_shape
        self.min_size = min_size
        self.max_ratio = max_ratio
        self.batch_size = batch_size

    def reset(self):
        self.iter.reset()

    def next(self):
        return self._process(self.iter.next())

    def iter_next(self):
        return self.iter.iter_next()

    def _process(self, batch):
        """Randomly create samples for detection."""
        self._data = mx.nd.zeros((self.batch_size, 1, self.data_shape, self.data_shape))
        self._label = mx.nd.full((self.batch_size, 3, 5), -1)
        for k in range(self.batch_size):
            w = int(round(random.uniform(self.min_size , 1) * self.data_shape))
            h = int(round(random.uniform(self.min_size , 1) * self.data_shape))
            if float(w) / h > self.max_ratio:
                w = int(round(h * self.max_ratio))
            if float(h) / w > self.max_ratio:
                h = int(round(w * self.max_ratio))
            warp = mx.image.imresize(batch.data[0][k].reshape((28, 28, 1)), w, h)
            x0 = random.randint(0, self.data_shape - w)
            y0 = random.randint(0, self.data_shape - h)
            self._data[k, 0, y0:y0+h, x0:x0+w] = warp.reshape((1, 1, h, w))
            width = height = float(self.data_shape)
            label = [batch.label[0][k].asscalar(), x0 / width, y0 / height,
                     (x0 + w) / width, (y0 + h) / height]
            self._label[k, 0, :] = mx.nd.array(label).reshape((1, 1, 5))
        self._batch = mx.io.DataBatch(data=[self._data], label=[self._label],
                                      index=batch.index, pad=batch.pad)
        return self._batch


def det_mnist_iterator(batch_size):
    curr_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
    sys.path.append(os.path.join(curr_path, "../../../tests/python/common"))
    import get_data
    get_data.GetMNIST_ubyte()

    train_iterator = DetMNISTIter(
        image="data/train-images-idx3-ubyte",
        label="data/train-labels-idx1-ubyte",
        batch_size=batch_size,
        min_size=0.3,
        shuffle=True)

    val_iterator = DetMNISTIter(
        image="data/t10k-images-idx3-ubyte",
        label="data/t10k-labels-idx1-ubyte",
        batch_size=batch_size,
        min_size=0.3,
        shuffle=False)

    return train_iterator, val_iterator, (1, 64, 64), [str(x) for x in range(10)]
