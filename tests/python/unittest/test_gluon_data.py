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
from mxnet import gluon

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


def test_recordimage_dataset():
    recfile = prepare_record()
    dataset = gluon.data.vision.ImageRecordDataset(recfile)
    loader = gluon.data.DataLoader(dataset, 1)

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

def test_image_folder_dataset():
    prepare_record()
    dataset = gluon.data.vision.ImageFolderDataset('data/test_images')
    assert dataset.synsets == ['test_images']
    assert len(dataset.items) == 16


class Dataset(gluon.data.Dataset):
    def __len__(self):
        return 100
    def __getitem__(self, key):
        return mx.nd.full((10,), key)

@unittest.skip("Somehow fails with MKL. Cannot reproduce locally")
def test_multi_worker():
    data = Dataset()
    loader = gluon.data.DataLoader(data, batch_size=1, num_workers=5)
    for i, batch in enumerate(loader):
        assert (batch.asnumpy() == i).all()


if __name__ == '__main__':
    import nose
    nose.runmodule()
