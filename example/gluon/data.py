# pylint: skip-file
""" data iterator for mnist """
import os
import random
import sys
# code to automatically download dataset
curr_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
sys.path.append(os.path.join(curr_path, "../../tests/python/common"))
import get_data
import mxnet as mx

def mnist_iterator(batch_size, input_shape):
    """return train and val iterators for mnist"""
    # download data
    get_data.GetMNIST_ubyte()
    flat = False if len(input_shape) == 3 else True

    train_dataiter = mx.io.MNISTIter(
        image="data/train-images-idx3-ubyte",
        label="data/train-labels-idx1-ubyte",
        input_shape=input_shape,
        batch_size=batch_size,
        shuffle=True,
        flat=flat)

    val_dataiter = mx.io.MNISTIter(
        image="data/t10k-images-idx3-ubyte",
        label="data/t10k-labels-idx1-ubyte",
        input_shape=input_shape,
        batch_size=batch_size,
        flat=flat)

    return (train_dataiter, val_dataiter)


def cifar10_iterator(batch_size, data_shape, resize=-1):
    get_data.GetCifar10()

    train = mx.io.ImageRecordIter(
        path_imgrec = "data/cifar/train.rec",
        # mean_img    = "data/cifar/mean.bin",
        resize      = resize,
        data_shape  = data_shape,
        batch_size  = batch_size,
        rand_crop   = True,
        rand_mirror = True)

    val = mx.io.ImageRecordIter(
        path_imgrec = "data/cifar/test.rec",
        # mean_img    = "data/cifar/mean.bin",
        resize      = resize,
        rand_crop   = False,
        rand_mirror = False,
        data_shape  = data_shape,
        batch_size  = batch_size)

    return train, val

class DummyIter(mx.io.DataIter):
    def __init__(self, batch_size, data_shape, batches = 5):
        super(DummyIter, self).__init__(batch_size)
        self.data_shape = (batch_size,) + data_shape
        self.label_shape = (batch_size,)
        self.provide_data = [('data', self.data_shape)]
        self.provide_label = [('softmax_label', self.label_shape)]
        self.batch = mx.io.DataBatch(data=[mx.nd.zeros(self.data_shape)],
                                     label=[mx.nd.zeros(self.label_shape)])
        self._batches = 0
        self.batches = batches

    def next(self):
        if self._batches < self.batches:
            self._batches += 1
            return self.batch
        else:
            self._batches = 0
            raise StopIteration

def dummy_iterator(batch_size, data_shape):
    return DummyIter(batch_size, data_shape), DummyIter(batch_size, data_shape)

class ImagePairIter(mx.io.DataIter):
    def __init__(self, path, data_shape, label_shape, batch_size=64, flag=0, input_aug=None, target_aug=None):
        super(ImagePairIter, self).__init__(batch_size)
        self.data_shape = (batch_size,) + data_shape
        self.label_shape = (batch_size,) + label_shape
        self.input_aug = input_aug
        self.target_aug = target_aug
        self.provide_data = [('data', self.data_shape)]
        self.provide_label = [('label', self.label_shape)]
        is_image_file = lambda fn: any(fn.endswith(ext) for ext in [".png", ".jpg", ".jpeg"])
        self.filenames = [os.path.join(path, x) for x in os.listdir(path) if is_image_file(x)]
        self.count = 0
        self.flag = flag
        random.shuffle(self.filenames)

    def next(self):
        from PIL import Image
        if self.count + self.batch_size <= len(self.filenames):
            data = []
            label = []
            for i in range(self.batch_size):
                fn = self.filenames[self.count]
                self.count += 1
                image = Image.open(fn).convert('YCbCr').split()[0]
                if image.size[0] > image.size[1]:
                    image = image.transpose(Image.TRANSPOSE)
                image = mx.nd.expand_dims(mx.nd.array(image), axis=2)
                target = image.copy()
                for aug in self.input_aug:
                    image = aug(image)[0]
                for aug in self.target_aug:
                    target = aug(target)[0]
                data.append(image)
                label.append(target)

            data = mx.nd.concat(*[mx.nd.expand_dims(d, axis=0) for d in data], dim=0)
            label = mx.nd.concat(*[mx.nd.expand_dims(d, axis=0) for d in label], dim=0)
            data = [mx.nd.transpose(data, axes=(0, 3, 1, 2)).astype('float32')/255]
            label = [mx.nd.transpose(label, axes=(0, 3, 1, 2)).astype('float32')/255]

            return mx.io.DataBatch(data=data, label=label)
        else:
            raise StopIteration

    def reset(self):
        self.count = 0
        random.shuffle(self.filenames)
