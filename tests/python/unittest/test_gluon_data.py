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
import tempfile
import unittest
import mxnet as mx
import numpy as np
import random
from mxnet import gluon
import platform
from mxnet.gluon.data import DataLoader
import mxnet.ndarray as nd
from mxnet import context
from mxnet.gluon.data.dataset import Dataset
from mxnet.gluon.data.dataset import ArrayDataset
import pytest

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

@pytest.fixture(scope="session")
def prepare_record(tmpdir_factory):
    test_images = tmpdir_factory.mktemp("test_images")
    test_images_tar = test_images.join("test_images.tar.gz")
    gluon.utils.download("https://repo.mxnet.io/gluon/dataset/test/test_images-9cebe48a.tar.gz", str(test_images_tar))
    tarfile.open(test_images_tar).extractall(str(test_images))
    imgs = os.listdir(str(test_images.join("test_images")))
    record = mx.recordio.MXIndexedRecordIO(str(test_images.join("test.idx")), str(test_images.join("test.rec")), 'w')
    for i, img in enumerate(imgs):
        with open(str(test_images.join("test_images").join(img)), 'rb') as f:
            str_img = f.read()
            s = mx.recordio.pack((0, i, i, 0), str_img)
            record.write_idx(i, s)
    return str(test_images.join('test.rec'))


def test_recordimage_dataset(prepare_record):
    recfile = prepare_record
    fn = lambda x, y : (x, y)
    dataset = gluon.data.vision.ImageRecordDataset(recfile).transform(fn)
    loader = gluon.data.DataLoader(dataset, 1)

    for i, (x, y) in enumerate(loader):
        assert x.shape[0] == 1 and x.shape[3] == 3
        assert y.asscalar() == i

@mx.util.use_np
def test_recordimage_dataset_handle(prepare_record):
    recfile = prepare_record
    class TmpTransform(mx.gluon.HybridBlock):
        def forward(self, x):
            return x

    fn = TmpTransform()
    dataset = gluon.data.vision.ImageRecordDataset(recfile).transform_first(fn).__mx_handle__()
    loader = gluon.data.DataLoader(dataset, 1)

    for i, (x, y) in enumerate(loader):
        assert x.shape[0] == 1 and x.shape[3] == 3
        assert y.item() == i

def _dataset_transform_fn(x, y):
    """Named transform function since lambda function cannot be pickled."""
    return x, y

def _dataset_transform_first_fn(x):
    """Named transform function since lambda function cannot be pickled."""
    return x

def test_recordimage_dataset_with_data_loader_multiworker(prepare_record):
    recfile = prepare_record
    dataset = gluon.data.vision.ImageRecordDataset(recfile)
    loader = gluon.data.DataLoader(dataset, 1, num_workers=5, try_nopython=False)

    for i, (x, y) in enumerate(loader):
        assert x.shape[0] == 1 and x.shape[3] == 3
        assert y.asscalar() == i

    # with transform
    dataset = gluon.data.vision.ImageRecordDataset(recfile).transform(_dataset_transform_fn)
    loader = gluon.data.DataLoader(dataset, 1, num_workers=5, try_nopython=None)

    for i, (x, y) in enumerate(loader):
        assert x.shape[0] == 1 and x.shape[3] == 3
        assert y.asscalar() == i

    # with transform_first
    dataset = gluon.data.vision.ImageRecordDataset(recfile).transform_first(_dataset_transform_first_fn)
    loader = gluon.data.DataLoader(dataset, 1, num_workers=5, try_nopython=None)

    for i, (x, y) in enumerate(loader):
        assert x.shape[0] == 1 and x.shape[3] == 3
        assert y.asscalar() == i

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

def test_datasets(tmpdir):
    p = tmpdir.mkdir("test_datasets")
    assert len(gluon.data.vision.MNIST(root=str(p.join('mnist')))) == 60000
    assert len(gluon.data.vision.MNIST(root=str(p.join('mnist')), train=False)) == 10000
    assert len(gluon.data.vision.FashionMNIST(root=str(p.join('fashion-mnist')))) == 60000
    assert len(gluon.data.vision.FashionMNIST(root=str(p.join('fashion-mnist')), train=False)) == 10000
    assert len(gluon.data.vision.CIFAR10(root=str(p.join('cifar10')))) == 50000
    assert len(gluon.data.vision.CIFAR10(root=str(p.join('cifar10')), train=False)) == 10000
    assert len(gluon.data.vision.CIFAR100(root=str(p.join('cifar100')))) == 50000
    assert len(gluon.data.vision.CIFAR100(root=str(p.join('cifar100')), fine_label=True)) == 50000
    assert len(gluon.data.vision.CIFAR100(root=str(p.join('cifar100')), train=False)) == 10000

def test_datasets_handles(tmpdir):
    p = tmpdir.mkdir("test_datasets_handles")
    assert len(gluon.data.vision.MNIST(root=str(p.join('mnist'))).__mx_handle__()) == 60000
    assert len(gluon.data.vision.MNIST(root=str(p.join('mnist')), train=False).__mx_handle__()) == 10000
    assert len(gluon.data.vision.FashionMNIST(root=str(p.join('fashion-mnist'))).__mx_handle__()) == 60000
    assert len(gluon.data.vision.FashionMNIST(root=str(p.join('fashion-mnist')), train=False).__mx_handle__()) == 10000
    assert len(gluon.data.vision.CIFAR10(root=str(p.join('cifar10'))).__mx_handle__()) == 50000
    assert len(gluon.data.vision.CIFAR10(root=str(p.join('cifar10')), train=False).__mx_handle__()) == 10000
    assert len(gluon.data.vision.CIFAR100(root=str(p.join('cifar100'))).__mx_handle__()) == 50000
    assert len(gluon.data.vision.CIFAR100(root=str(p.join('cifar100')), fine_label=True).__mx_handle__()) == 50000
    assert len(gluon.data.vision.CIFAR100(root=str(p.join('cifar100')), train=False).__mx_handle__()) == 10000

def test_image_folder_dataset(prepare_record):
    dataset = gluon.data.vision.ImageFolderDataset(os.path.dirname(prepare_record))
    assert dataset.synsets == ['test_images']
    assert len(dataset.items) == 16

def test_image_folder_dataset_handle(prepare_record):
    dataset = gluon.data.vision.ImageFolderDataset(os.path.dirname(prepare_record))
    hd = dataset.__mx_handle__()
    assert len(hd) == 16
    assert (hd[1][0] == dataset[1][0]).asnumpy().all()
    assert hd[5][1] == dataset[5][1]

def test_image_list_dataset(prepare_record):
    root = os.path.join(os.path.dirname(prepare_record), 'test_images')
    imlist = os.listdir(root)
    imglist = [(0, path) for i, path in enumerate(imlist)]
    dataset = gluon.data.vision.ImageListDataset(root=root, imglist=imglist)
    assert len(dataset) == 16, len(dataset)
    img, label = dataset[0]
    assert len(img.shape) == 3
    assert label == 0

    # save to file as *.lst
    imglist = ['\t'.join((str(i), '0', path)) for i, path in enumerate(imlist)]
    with tempfile.NamedTemporaryFile('wt', delete=False) as fp:
        for line in imglist:
            fp.write(line + '\n')
        fp.close()

        dataset = gluon.data.vision.ImageListDataset(root=root, imglist=fp.name)
        assert len(dataset) == 16, len(dataset)
        img, label = dataset[0]
        assert len(img.shape) == 3
        assert label == 0

def test_image_list_dataset_handle(prepare_record):
    root = os.path.join(os.path.dirname(prepare_record), 'test_images')
    imlist = os.listdir(root)
    imglist = [(0, path) for i, path in enumerate(imlist)]
    dataset = gluon.data.vision.ImageListDataset(root=root, imglist=imglist).__mx_handle__()
    assert len(dataset) == 16, len(dataset)
    img, label = dataset[0]
    assert len(img.shape) == 3
    assert label == 0

    # save to file as *.lst
    imglist = ['\t'.join((str(i), '0', path)) for i, path in enumerate(imlist)]
    with tempfile.NamedTemporaryFile('wt', delete=False) as fp:
        for line in imglist:
            fp.write(line + '\n')
        fp.close()

        dataset = gluon.data.vision.ImageListDataset(root=root, imglist=fp.name).__mx_handle__()
        assert len(dataset) == 16
        img, label = dataset[0]
        assert len(img.shape) == 3
        assert label == 0

@pytest.mark.garbage_expected
def test_list_dataset():
    for num_worker in range(0, 3):
        data = mx.gluon.data.DataLoader([([1,2], 0), ([3, 4], 1)], batch_size=1, num_workers=num_worker)
        for d, l in data:
            pass


class _Dataset(gluon.data.Dataset):
    def __len__(self):
        return 100
    def __getitem__(self, key):
        return mx.nd.full((10,), key)

@pytest.mark.garbage_expected
def test_multi_worker():
    data = _Dataset()
    for thread_pool in [True, False]:
        loader = gluon.data.DataLoader(data, batch_size=1, num_workers=5, thread_pool=thread_pool)
        for i, batch in enumerate(loader):
            assert (batch.asnumpy() == i).all()


def test_multi_worker_shape():
    for thread_pool in [True, False]:
        batch_size = 1024
        shape = (batch_size+1, 11, 12)

        data = ArrayDataset(np.ones(shape))
        loader = gluon.data.DataLoader(
            data, batch_size=batch_size, num_workers=5, last_batch='keep', thread_pool=thread_pool)
        for batch in loader:
            if shape[0] > batch_size:
                assert batch.shape == (batch_size, shape[1], shape[2])
                shape = (shape[0] - batch_size, shape[1], shape[2])
            else:
                assert batch.shape == shape

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

def test_multi_worker_dataloader_release_pool():
    # will trigger too many open file if pool is not released properly
    if os.name == 'nt':
        print('Skip for windows since spawn on windows is too expensive.')
        return

    for _ in range(10):
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

    if mx.context.num_gpus() <= 1:
        print('Bypassing custom_dev_id pinned mem test on system with < 2 gpus.')
    else:
        # use pinned memory with custom device id
        loader3 = gluon.data.DataLoader(dataset, 8, pin_memory=True,
                                        pin_device_id=custom_dev_id)
        for _, x in enumerate(loader3):
            assert x.context == context.cpu_pinned(custom_dev_id)

def batchify(a):
    return a

def test_dataset_filter():
    length = 100
    a = mx.gluon.data.SimpleDataset([i for i in range(length)])
    a_filtered = a.filter(lambda x: x % 10 == 0)
    assert(len(a_filtered) == 10)
    for idx, sample in enumerate(a_filtered):
        assert sample % 10 == 0
    a_xform_filtered = a.transform(lambda x: x + 1).filter(lambda x: x % 10 == 0)
    assert(len(a_xform_filtered) == 10)
    # the filtered data is already transformed
    for idx, sample in enumerate(a_xform_filtered):
        assert sample % 10 == 0

def test_dataset_filter_handle():
    length = 100
    a = mx.gluon.data.SimpleDataset(np.arange(length))
    a_filtered = a.filter(lambda x: x % 10 == 0).__mx_handle__()
    assert(len(a_filtered) == 10)
    for idx, sample in enumerate(a_filtered):
        assert sample % 10 == 0
    a_xform_filtered = a.transform(lambda x: x + 1).filter(lambda x: x % 10 == 0)
    assert(len(a_xform_filtered) == 10)
    # the filtered data is already transformed
    for idx, sample in enumerate(a_xform_filtered):
        assert sample % 10 == 0

def test_dataset_shard():
    length = 9
    a = mx.gluon.data.SimpleDataset([i for i in range(length)])
    shard_0 = a.shard(4, 0)
    shard_1 = a.shard(4, 1)
    shard_2 = a.shard(4, 2)
    shard_3 = a.shard(4, 3)
    assert len(shard_0) + len(shard_1) + len(shard_2) + len(shard_3) == length
    assert len(shard_0) == 3
    assert len(shard_1) == 2
    assert len(shard_2) == 2
    assert len(shard_3) == 2
    total = 0
    for shard in [shard_0, shard_1, shard_2, shard_3]:
        for idx, sample in enumerate(shard):
            total += sample
    assert total == sum(a)

def test_dataset_shard_handle():
    length = 9
    a = mx.gluon.data.SimpleDataset(np.arange(length))
    shard_0 = a.shard(4, 0).__mx_handle__()
    shard_1 = a.shard(4, 1).__mx_handle__()
    shard_2 = a.shard(4, 2).__mx_handle__()
    shard_3 = a.shard(4, 3).__mx_handle__()
    assert len(shard_0) + len(shard_1) + len(shard_2) + len(shard_3) == length
    assert len(shard_0) == 3
    assert len(shard_1) == 2
    assert len(shard_2) == 2
    assert len(shard_3) == 2
    total = 0
    for shard in [shard_0, shard_1, shard_2, shard_3]:
        for idx, sample in enumerate(shard):
            total += sample
    assert total == sum(a)

def test_dataset_take():
    length = 100
    a = mx.gluon.data.SimpleDataset([i for i in range(length)])
    a_take_full = a.take(1000)
    assert len(a_take_full) == length
    a_take_full = a.take(None)
    assert len(a_take_full) == length
    count = 10
    a_take_10 = a.take(count)
    assert len(a_take_10) == count
    expected_total = sum([i for i in range(count)])
    total = 0
    for idx, sample in enumerate(a_take_10):
        assert sample < count
        total += sample
    assert total == expected_total

    a_xform_take_10 = a.transform(lambda x: x * 10).take(count)
    assert len(a_xform_take_10) == count
    expected_total = sum([i * 10 for i in range(count)])
    total = 0
    for idx, sample in enumerate(a_xform_take_10):
        assert sample < count * 10
        total += sample
    assert total == expected_total

def test_dataset_take_handle():
    length = 100
    a = mx.gluon.data.SimpleDataset(np.arange(length))
    a_take_full = a.take(1000).__mx_handle__()
    assert len(a_take_full) == length
    a_take_full = a.take(None).__mx_handle__()
    assert len(a_take_full) == length
    count = 10
    a_take_10 = a.take(count).__mx_handle__()
    assert len(a_take_10) == count
    expected_total = sum([i for i in range(count)])
    total = 0
    for idx, sample in enumerate(a_take_10):
        assert sample < count
        total += sample
    assert total == expected_total

    a_xform_take_10 = a.take(count).__mx_handle__()
    assert len(a_xform_take_10) == count
    expected_total = sum([i for i in range(count)])
    total = 0
    for idx, sample in enumerate(a_xform_take_10):
        assert sample < count
        total += sample
    assert total == expected_total

@pytest.mark.garbage_expected
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

def test_mx_datasets_handle():
    # _DownloadedDataset
    mnist = mx.gluon.data.vision.MNIST(train=False).__mx_handle__()
    assert len(mnist) == 10000
    cifar10 = mx.gluon.data.vision.CIFAR10(train=False).__mx_handle__()
    assert len(cifar10) == 10000

    # _SampledDataset
    s_mnist = mnist.take(100).__mx_handle__()
    assert len(s_mnist) == 100
    assert np.all(s_mnist[0][0].asnumpy() == mnist[0][0].asnumpy())
    assert s_mnist[0][1] == mnist[0][1]

    # ArrayDataset
    mc = mx.gluon.data.ArrayDataset(mnist.take(100), cifar10.take(100)).__mx_handle__()
    assert len(mc) == 100
    assert len(mc[0]) == 4  # two from mnist, two from cifar10
    assert mc[0][1] == mnist[0][1]
    assert mc[0][3] == cifar10[0][1]

def test_mx_data_loader():
    from mxnet.gluon.data.dataloader import DataLoader

    dataset = mx.gluon.data.vision.MNIST(train=False)
    dl = DataLoader(num_workers=0, dataset=dataset, batch_size=32)
    for _ in dl:
        pass

@mx.util.use_np
def test_mx_data_loader_nopython():
    from mxnet.gluon.data.dataloader import DataLoader
    from mxnet.gluon.data.vision.transforms import ToTensor
    dataset = mx.gluon.data.vision.MNIST(train=False)
    dl1 = DataLoader(dataset=dataset.transform_first(ToTensor()), batch_size=32, try_nopython=True, shuffle=False)
    dl2 = DataLoader(dataset=dataset.transform_first(ToTensor()), batch_size=32, try_nopython=False, shuffle=False)
    assert len(dl1) == len(dl2)
    assert np.all(next(iter(dl1))[1].asnumpy() == next(iter(dl2))[1].asnumpy())
    for _ in dl1:
        pass

def test_batchify_stack():
    a = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
    b = np.array([[5, 6, 7, 8], [1, 2, 3, 4]])
    bf = mx.gluon.data.batchify.Stack()
    bf_handle = bf.__mx_handle__()
    c = bf([a, b])
    d = bf_handle([a, b])
    assert c.shape == d.shape
    assert mx.test_utils.almost_equal(c.asnumpy(), d.asnumpy())
    assert mx.test_utils.almost_equal(c.asnumpy(), np.stack((a, b)))

def test_batchify_pad():
    a = np.array([[1, 2, 3, 4], [11, 12, 13, 14]])
    b = np.array([[4, 5, 6]])
    c = np.array([[9, 10]])
    bf = mx.gluon.data.batchify.Pad(val=-1)
    bf_handle = bf.__mx_handle__()
    d = bf([a, b, c])
    e = bf_handle([a, b, c])
    assert d.shape == e.shape
    assert mx.test_utils.almost_equal(d.asnumpy(), e.asnumpy())
    expected = np.array([[[ 1.,  2.,  3.,  4.], [11., 12., 13., 14.]],
                         [[ 4.,  5.,  6., -1.], [-1., -1., -1., -1.]],
                         [[ 9., 10., -1., -1.], [-1., -1., -1., -1.]]])
    assert mx.test_utils.almost_equal(d.asnumpy(), expected)

def test_batchify_group():
    a = [np.array([[1, 2, 3, 4], [5, 6, 7, 8]]), np.array([[1, 2, 3, 4], [11, 12, 13, 14]])]
    b = [np.array([[1, 2, 3, 4], [5, 6, 7, 8]]), np.array([[4, 5, 6]])]
    c = [np.array([[1, 2, 3, 4], [5, 6, 7, 8]]), np.array([[9, 10]])]
    bf = mx.gluon.data.batchify.Group(mx.gluon.data.batchify.Stack(), mx.gluon.data.batchify.Pad(val=-1))
    bf_handle = bf.__mx_handle__()
    d = bf([a, b, c])
    e = bf_handle([a, b, c])
    assert d[0].shape == e[0].shape
    assert d[1].shape == e[1].shape
    print(d[0].asnumpy(), ',', e[0].asnumpy(), ',', e[1].asnumpy())
    assert mx.test_utils.almost_equal(d[0].asnumpy(), e[0].asnumpy())
    assert mx.test_utils.almost_equal(d[1].asnumpy(), e[1].asnumpy())
    assert mx.test_utils.almost_equal(d[0].asnumpy(), np.stack((a[0], b[0], c[0])))
    expected = np.array([[[ 1.,  2.,  3.,  4.], [11., 12., 13., 14.]],
                         [[ 4.,  5.,  6., -1.], [-1., -1., -1., -1.]],
                         [[ 9., 10., -1., -1.], [-1., -1., -1., -1.]]])
    assert mx.test_utils.almost_equal(d[1].asnumpy(), expected)

def test_sampler():
    interval_sampler = mx.gluon.data.IntervalSampler(10, 3)
    assert sorted(list(interval_sampler)) == list(range(10))
    interval_sampler = mx.gluon.data.IntervalSampler(10, 3, rollover=False)
    assert list(interval_sampler) == [0, 3, 6, 9]
