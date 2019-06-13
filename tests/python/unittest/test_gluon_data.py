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

import os
import tarfile
import unittest
import mxnet as mx
import numpy as np
import random
from mxnet import gluon
import platform
from common import setup_module, with_seed, teardown
from mxnet.gluon.data import DataLoader
import mxnet.ndarray as nd
from mxnet import context
from mxnet.gluon.data.dataset import Dataset
from mxnet.gluon.data.dataset import ArrayDataset

@with_seed()
def test_array_dataset():
    X = np.random.uniform(size=(10, 20))
    Y = np.random.uniform(size=(10,))
    dataset = gluon.data.ArrayDataset(X, Y)
    loader = gluon.data.DataLoader(dataset, 2)
    for i, (x, y) in enumerate(loader):
        assert mx.test_utils.almost_equal(x.asnumpy(), X[i*2:(i+1)*2])
        assert mx.test_utils.almost_equal(y.asnumpy(), Y[i*2:(i+1)*2])

    dataset = gluon.data.ArrayDataset(X)
    loader = gluon.data.DataLoader(dataset, 2)

    for i, x in enumerate(loader):
        assert mx.test_utils.almost_equal(x.asnumpy(), X[i*2:(i+1)*2])


def prepare_record():
    if not os.path.isdir("data/test_images"):
        os.makedirs('data/test_images')
    if not os.path.isdir("data/test_images/test_images"):
        gluon.utils.download("http://data.mxnet.io/data/test_images.tar.gz", "data/test_images.tar.gz")
        tarfile.open('data/test_images.tar.gz').extractall('data/test_images/')
    if not os.path.exists('data/test.rec'):
        imgs = os.listdir('data/test_images/test_images')
        record = mx.recordio.MXIndexedRecordIO('data/test.idx', 'data/test.rec', 'w')
        for i, img in enumerate(imgs):
            str_img = open('data/test_images/test_images/'+img, 'rb').read()
            s = mx.recordio.pack((0, i, i, 0), str_img)
            record.write_idx(i, s)
    return 'data/test.rec'


@with_seed()
def test_recordimage_dataset():
    recfile = prepare_record()
    fn = lambda x, y : (x, y)
    dataset = gluon.data.vision.ImageRecordDataset(recfile).transform(fn)
    loader = gluon.data.DataLoader(dataset, 1)

    for i, (x, y) in enumerate(loader):
        assert x.shape[0] == 1 and x.shape[3] == 3
        assert y.asscalar() == i

def _dataset_transform_fn(x, y):
    """Named transform function since lambda function cannot be pickled."""
    return x, y

def _dataset_transform_first_fn(x):
    """Named transform function since lambda function cannot be pickled."""
    return x

@with_seed()
def test_recordimage_dataset_with_data_loader_multiworker():
    recfile = prepare_record()
    dataset = gluon.data.vision.ImageRecordDataset(recfile)
    loader = gluon.data.DataLoader(dataset, 1, num_workers=5)

    for i, (x, y) in enumerate(loader):
        assert x.shape[0] == 1 and x.shape[3] == 3
        assert y.asscalar() == i

    # with transform
    dataset = gluon.data.vision.ImageRecordDataset(recfile).transform(_dataset_transform_fn)
    loader = gluon.data.DataLoader(dataset, 1, num_workers=5)

    for i, (x, y) in enumerate(loader):
        assert x.shape[0] == 1 and x.shape[3] == 3
        assert y.asscalar() == i

    # with transform_first
    dataset = gluon.data.vision.ImageRecordDataset(recfile).transform_first(_dataset_transform_first_fn)
    loader = gluon.data.DataLoader(dataset, 1, num_workers=5)

    for i, (x, y) in enumerate(loader):
        assert x.shape[0] == 1 and x.shape[3] == 3
        assert y.asscalar() == i

@with_seed()
def test_sampler():
    seq_sampler = gluon.data.SequentialSampler(10)
    assert list(seq_sampler) == list(range(10))
    rand_sampler = gluon.data.RandomSampler(10)
    assert sorted(list(rand_sampler)) == list(range(10))
    seq_batch_keep = gluon.data.BatchSampler(seq_sampler, 3, 'keep')
    assert sum(list(seq_batch_keep), []) == list(range(10))
    seq_batch_discard = gluon.data.BatchSampler(seq_sampler, 3, 'discard')
    assert sum(list(seq_batch_discard), []) == list(range(9))
    rand_batch_keep = gluon.data.BatchSampler(rand_sampler, 3, 'keep')
    assert sorted(sum(list(rand_batch_keep), [])) == list(range(10))

@with_seed()
def test_datasets():
    assert len(gluon.data.vision.MNIST(root='data/mnist')) == 60000
    assert len(gluon.data.vision.MNIST(root='data/mnist', train=False)) == 10000
    assert len(gluon.data.vision.FashionMNIST(root='data/fashion-mnist')) == 60000
    assert len(gluon.data.vision.FashionMNIST(root='data/fashion-mnist', train=False)) == 10000
    assert len(gluon.data.vision.CIFAR10(root='data/cifar10')) == 50000
    assert len(gluon.data.vision.CIFAR10(root='data/cifar10', train=False)) == 10000
    assert len(gluon.data.vision.CIFAR100(root='data/cifar100')) == 50000
    assert len(gluon.data.vision.CIFAR100(root='data/cifar100', fine_label=True)) == 50000
    assert len(gluon.data.vision.CIFAR100(root='data/cifar100', train=False)) == 10000

@with_seed()
def test_image_folder_dataset():
    prepare_record()
    dataset = gluon.data.vision.ImageFolderDataset('data/test_images')
    assert dataset.synsets == ['test_images']
    assert len(dataset.items) == 16

@with_seed()
def test_list_dataset():
    for num_worker in range(0, 3):
        data = mx.gluon.data.DataLoader([([1,2], 0), ([3, 4], 1)], batch_size=1, num_workers=num_worker)
        for d, l in data:
            pass


class Dataset(gluon.data.Dataset):
    def __len__(self):
        return 100
    def __getitem__(self, key):
        return mx.nd.full((10,), key)

@with_seed()
def test_multi_worker():
    data = Dataset()
    for thread_pool in [True, False]:
        loader = gluon.data.DataLoader(data, batch_size=1, num_workers=5, thread_pool=thread_pool)
        for i, batch in enumerate(loader):
            assert (batch.asnumpy() == i).all()

class _Dummy(Dataset):
    """Dummy dataset for randomized shape arrays."""
    def __init__(self, random_shape):
        self.random_shape = random_shape

    def __getitem__(self, idx):
        key = idx
        if self.random_shape:
            out = np.random.uniform(size=(random.randint(1000, 1100), 40))
            labels = np.random.uniform(size=(random.randint(10, 15)))
        else:
            out = np.random.uniform(size=(1000, 40))
            labels = np.random.uniform(size=(10))
        return key, out, labels

    def __len__(self):
        return 50

def _batchify_list(data):
    """
    return list of ndarray without stack/concat/pad
    """
    if isinstance(data, (tuple, list)):
        return list(data)
    if isinstance(data, mx.nd.NDArray):
        return [data]
    return data

def _batchify(data):
    """
    Collate data into batch. Use shared memory for stacking.
    :param data: a list of array, with layout of 'NTC'.
    :return either x  and x's unpadded lengths, or x, x's unpadded lengths, y and y's unpadded lengths
            if labels are not supplied.
    """

    # input layout is NTC
    keys, inputs, labels = [item[0] for item in data], [item[1] for item in data], \
                           [item[2] for item in data]

    if len(data) > 1:
        max_data_len = max([seq.shape[0] for seq in inputs])
        max_labels_len = 0 if not labels else max([seq.shape[0] for seq in labels])
    else:
        max_data_len = inputs[0].shape[0]
        max_labels_len = 0 if not labels else labels[0].shape[0]

    x_lens = [item.shape[0] for item in inputs]
    y_lens = [item.shape[0] for item in labels]

    for i, seq in enumerate(inputs):
        pad_len = max_data_len - seq.shape[0]
        inputs[i] = np.pad(seq, ((0, pad_len), (0, 0)), 'constant', constant_values=0)
        labels[i] = np.pad(labels[i], (0, max_labels_len - labels[i].shape[0]),
                           'constant', constant_values=-1)

    inputs = np.asarray(inputs, dtype=np.float32)
    if labels is not None:
        labels = np.asarray(labels, dtype=np.float32)
    inputs = inputs.transpose((1, 0, 2))
    labels = labels.transpose((1, 0))

    return (nd.array(inputs, dtype=inputs.dtype, ctx=context.Context('cpu_shared', 0)),
            nd.array(x_lens, ctx=context.Context('cpu_shared', 0))) \
        if labels is None else (
        nd.array(inputs, dtype=inputs.dtype, ctx=context.Context('cpu_shared', 0)),
        nd.array(x_lens, ctx=context.Context('cpu_shared', 0)),
        nd.array(labels, dtype=labels.dtype, ctx=context.Context('cpu_shared', 0)),
        nd.array(y_lens, ctx=context.Context('cpu_shared', 0)))

@with_seed()
def test_multi_worker_forked_data_loader():
    data = _Dummy(False)
    loader = DataLoader(data, batch_size=40, batchify_fn=_batchify, num_workers=2)
    for epoch in range(1):
        for i, data in enumerate(loader):
            pass

    data = _Dummy(True)
    loader = DataLoader(data, batch_size=40, batchify_fn=_batchify_list, num_workers=2)
    for epoch in range(1):
        for i, data in enumerate(loader):
            pass

@with_seed()
def test_multi_worker_dataloader_release_pool():
    # will trigger too many open file if pool is not released properly
    for _ in range(100):
        A = np.random.rand(999, 2000)
        D = mx.gluon.data.DataLoader(A, batch_size=8, num_workers=8)
        the_iter = iter(D)
        next(the_iter)
        del the_iter
        del D


def test_dataloader_context():
    X = np.random.uniform(size=(10, 20))
    dataset = gluon.data.ArrayDataset(X)
    default_dev_id = 0
    custom_dev_id = 1

    # use non-pinned memory
    loader1 = gluon.data.DataLoader(dataset, 8)
    for _, x in enumerate(loader1):
        assert x.context == context.cpu(default_dev_id)

    # use pinned memory with default device id
    loader2 = gluon.data.DataLoader(dataset, 8, pin_memory=True)
    for _, x in enumerate(loader2):
        assert x.context == context.cpu_pinned(default_dev_id)

    # use pinned memory with custom device id
    loader3 = gluon.data.DataLoader(dataset, 8, pin_memory=True,
                                    pin_device_id=custom_dev_id)
    for _, x in enumerate(loader3):
        assert x.context == context.cpu_pinned(custom_dev_id)

def batchify(a):
    return a

def test_dataloader_scope():
    """
    Bug: Gluon DataLoader terminates the process pool early while
    _MultiWorkerIter is operating on the pool.

    Tests that DataLoader is not garbage collected while the iterator is
    in use.
    """
    args = {'num_workers': 1, 'batch_size': 2}
    dataset = nd.ones(5)
    iterator = iter(DataLoader(
            dataset,
            batchify_fn=batchify,
            **args
        )
    )

    item = next(iterator)

    assert item is not None


if __name__ == '__main__':
    import nose
    nose.runmodule()
