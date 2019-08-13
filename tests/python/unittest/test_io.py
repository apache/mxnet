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

# pylint: skip-file
import mxnet as mx
import mxnet.ndarray as nd
from mxnet.test_utils import *
from mxnet.base import MXNetError
import numpy as np
import os
import gzip
import pickle as pickle
import time
try:
    import h5py
except ImportError:
    h5py = None
import sys
from common import assertRaises
import unittest
try:
    from itertools import izip_longest as zip_longest
except:
    from itertools import zip_longest


def test_MNISTIter():
    # prepare data
    get_mnist_ubyte()

    batch_size = 100
    train_dataiter = mx.io.MNISTIter(
        image="data/train-images-idx3-ubyte",
        label="data/train-labels-idx1-ubyte",
        data_shape=(784,),
        batch_size=batch_size, shuffle=1, flat=1, silent=0, seed=10)
    # test_loop
    nbatch = 60000 / batch_size
    batch_count = 0
    for batch in train_dataiter:
        batch_count += 1
    assert(nbatch == batch_count)
    # test_reset
    train_dataiter.reset()
    train_dataiter.iter_next()
    label_0 = train_dataiter.getlabel().asnumpy().flatten()
    train_dataiter.iter_next()
    train_dataiter.iter_next()
    train_dataiter.iter_next()
    train_dataiter.iter_next()
    train_dataiter.reset()
    train_dataiter.iter_next()
    label_1 = train_dataiter.getlabel().asnumpy().flatten()
    assert(sum(label_0 - label_1) == 0)


def test_Cifar10Rec():
    get_cifar10()
    dataiter = mx.io.ImageRecordIter(
        path_imgrec="data/cifar/train.rec",
        mean_img="data/cifar/cifar10_mean.bin",
        rand_crop=False,
        and_mirror=False,
        shuffle=False,
        data_shape=(3, 28, 28),
        batch_size=100,
        preprocess_threads=4,
        prefetch_buffer=1)
    labelcount = [0 for i in range(10)]
    batchcount = 0
    for batch in dataiter:
        npdata = batch.data[0].asnumpy().flatten().sum()
        sys.stdout.flush()
        batchcount += 1
        nplabel = batch.label[0].asnumpy()
        for i in range(nplabel.shape[0]):
            labelcount[int(nplabel[i])] += 1
    for i in range(10):
        assert(labelcount[i] == 5000)

def test_inter_methods_in_augmenter():
    def test_Cifar10Rec():
        get_cifar10()
        for inter_method in [0,1,2,3,4,9,10]:
            dataiter = mx.io.ImageRecordIter(
                path_imgrec="data/cifar/train.rec",
                mean_img="data/cifar/cifar10_mean.bin",
                max_rotate_angle=45,
                inter_method=inter_method)
            for batch in dataiter:
                pass

def test_image_iter_exception():
    def check_cifar10_exception():
        get_cifar10()
        dataiter = mx.io.ImageRecordIter(
            path_imgrec="data/cifar/train.rec",
            mean_img="data/cifar/cifar10_mean.bin",
            rand_crop=False,
            and_mirror=False,
            shuffle=False,
            data_shape=(5, 28, 28),
            batch_size=100,
            preprocess_threads=4,
            prefetch_buffer=1)
        labelcount = [0 for i in range(10)]
        batchcount = 0
        for batch in dataiter:
            pass
    assertRaises(MXNetError, check_cifar10_exception)

def _init_NDArrayIter_data(data_type, is_image=False):
    if is_image:
        data = nd.random.uniform(0, 255, shape=(5000, 1, 28, 28))
        labels = nd.ones((5000, 1))
        return data, labels
    if data_type == 'NDArray':
        data = nd.ones((1000, 2, 2))
        labels = nd.ones((1000, 1))
    else:
        data = np.ones((1000, 2, 2))
        labels = np.ones((1000, 1))
    for i in range(1000):
        data[i] = i / 100
        labels[i] = i / 100
    return data, labels


def _test_last_batch_handle(data, labels=None, is_image=False):
    # Test the three parameters 'pad', 'discard', 'roll_over'
    last_batch_handle_list = ['pad', 'discard', 'roll_over']
    if labels is not None and not is_image and len(labels) != 0:
        labelcount_list = [(124, 100), (100, 96), (100, 96)]
    if is_image:
        batch_count_list = [40, 39, 39]
    else:
        batch_count_list = [8, 7, 7]
    
    for idx in range(len(last_batch_handle_list)):
        dataiter = mx.io.NDArrayIter(
            data, labels, 128, False, last_batch_handle=last_batch_handle_list[idx])
        batch_count = 0
        if labels is not None and len(labels) != 0 and not is_image:
            labelcount = [0 for i in range(10)]
        for batch in dataiter:
            if len(data) == 2:
                assert len(batch.data) == 2
            if labels is not None and len(labels) != 0:
                if not is_image:
                    label = batch.label[0].asnumpy().flatten()
                    # check data if it matches corresponding labels
                    assert((batch.data[0].asnumpy()[:, 0, 0] == label).all())
                    for i in range(label.shape[0]):
                       labelcount[int(label[i])] += 1
            else:
                assert not batch.label, 'label is not empty list'
            # keep the last batch of 'pad' to be used later 
            # to test first batch of roll_over in second iteration
            batch_count += 1
            if last_batch_handle_list[idx] == 'pad' and \
                batch_count == batch_count_list[0]:
                cache = batch.data[0].asnumpy()
        # check if batchifying functionality work properly
        if labels is not None and len(labels) != 0 and not is_image:
            assert labelcount[0] == labelcount_list[idx][0], last_batch_handle_list[idx]
            assert labelcount[8] == labelcount_list[idx][1], last_batch_handle_list[idx]
        assert batch_count == batch_count_list[idx]
    # roll_over option
    dataiter.reset()
    assert np.array_equal(dataiter.next().data[0].asnumpy(), cache)


def _test_shuffle(data, labels=None):
    dataiter = mx.io.NDArrayIter(data, labels, 1, False)
    batch_list = []
    for batch in dataiter:
        # cache the original data
        batch_list.append(batch.data[0].asnumpy())
    dataiter = mx.io.NDArrayIter(data, labels, 1, True)
    idx_list = dataiter.idx
    i = 0
    for batch in dataiter:
        # check if each data point have been shuffled to corresponding positions
        assert np.array_equal(batch.data[0].asnumpy(), batch_list[idx_list[i]])
        i += 1


def test_NDArrayIter():
    dtype_list = ['NDArray', 'ndarray']
    tested_data_type = [False, True]
    for dtype in dtype_list:
        for is_image in tested_data_type:
            data, labels = _init_NDArrayIter_data(dtype, is_image)
            _test_last_batch_handle(data, labels, is_image)
            _test_last_batch_handle([data, data], labels, is_image)
            _test_last_batch_handle(data=[data, data], is_image=is_image)
            _test_last_batch_handle(
                {'data1': data, 'data2': data}, labels, is_image)
            _test_last_batch_handle(data={'data1': data, 'data2': data}, is_image=is_image)
            _test_last_batch_handle(data, [], is_image)
            _test_last_batch_handle(data=data, is_image=is_image)
            _test_shuffle(data, labels)
            _test_shuffle([data, data], labels)
            _test_shuffle([data, data])
            _test_shuffle({'data1': data, 'data2': data}, labels)
            _test_shuffle({'data1': data, 'data2': data})
            _test_shuffle(data, [])
            _test_shuffle(data)


def test_NDArrayIter_h5py():
    if not h5py:
        return

    data, labels = _init_NDArrayIter_data('ndarray')

    try:
        os.remove('ndarraytest.h5')
    except OSError:
        pass
    with h5py.File('ndarraytest.h5') as f:
        f.create_dataset('data', data=data)
        f.create_dataset('label', data=labels)
        
        _test_last_batch_handle(f['data'], f['label'])
        _test_last_batch_handle(f['data'], [])
        _test_last_batch_handle(f['data'])
    try:
        os.remove("ndarraytest.h5")
    except OSError:
        pass


def _test_NDArrayIter_csr(csr_iter, csr_iter_empty_list, csr_iter_None, num_rows, batch_size):
    num_batch = 0
    for _, batch_empty_list, batch_empty_None in zip(csr_iter, csr_iter_empty_list, csr_iter_None):
        assert not batch_empty_list.label, 'label is not empty list'
        assert not batch_empty_None.label, 'label is not empty list'
        num_batch += 1

    assert(num_batch == num_rows // batch_size)
    assertRaises(StopIteration, csr_iter.next)
    assertRaises(StopIteration, csr_iter_empty_list.next)
    assertRaises(StopIteration, csr_iter_None.next)


def test_NDArrayIter_csr():
    # creating toy data
    num_rows = rnd.randint(5, 15)
    num_cols = rnd.randint(1, 20)
    batch_size = rnd.randint(1, num_rows)
    shape = (num_rows, num_cols)
    csr, _ = rand_sparse_ndarray(shape, 'csr')
    dns = csr.asnumpy()

    # CSRNDArray or scipy.sparse.csr_matrix with last_batch_handle not equal to 'discard' will throw NotImplementedError
    assertRaises(NotImplementedError, mx.io.NDArrayIter,
                 {'data': csr}, dns, batch_size)
    try:
        import scipy.sparse as spsp
        train_data = spsp.csr_matrix(dns)
        assertRaises(NotImplementedError, mx.io.NDArrayIter,
                     {'data': train_data}, dns, batch_size)
    except ImportError:
        pass
    
    # scipy.sparse.csr_matrix with shuffle
    csr_iter = iter(mx.io.NDArrayIter({'data': train_data}, dns, batch_size,
                                      shuffle=True, last_batch_handle='discard'))
    csr_iter_empty_list = iter(mx.io.NDArrayIter({'data': train_data}, [], batch_size,
                                      shuffle=True, last_batch_handle='discard'))
    csr_iter_None = iter(mx.io.NDArrayIter({'data': train_data}, None, batch_size,
                                      shuffle=True, last_batch_handle='discard'))
    _test_NDArrayIter_csr(csr_iter, csr_iter_empty_list,
                          csr_iter_None, num_rows, batch_size)

    # CSRNDArray with shuffle
    csr_iter = iter(mx.io.NDArrayIter({'csr_data': csr, 'dns_data': dns}, dns, batch_size,
                                      shuffle=True, last_batch_handle='discard'))
    csr_iter_empty_list = iter(mx.io.NDArrayIter({'csr_data': csr, 'dns_data': dns}, [], batch_size,
                                      shuffle=True, last_batch_handle='discard'))
    csr_iter_None = iter(mx.io.NDArrayIter({'csr_data': csr, 'dns_data': dns}, None, batch_size,
                                      shuffle=True, last_batch_handle='discard'))
    _test_NDArrayIter_csr(csr_iter, csr_iter_empty_list,
                          csr_iter_None, num_rows, batch_size)

    # make iterators
    csr_iter = iter(mx.io.NDArrayIter(
        csr, csr, batch_size, last_batch_handle='discard'))
    begin = 0
    for batch in csr_iter:
        expected = np.zeros((batch_size, num_cols))
        end = begin + batch_size
        expected[:num_rows - begin] = dns[begin:end]
        if end > num_rows:
            expected[num_rows - begin:] = dns[0:end - num_rows]
        assert_almost_equal(batch.data[0].asnumpy(), expected)
        begin += batch_size


def test_LibSVMIter():

    def check_libSVMIter_synthetic():
        cwd = os.getcwd()
        data_path = os.path.join(cwd, 'data.t')
        label_path = os.path.join(cwd, 'label.t')
        with open(data_path, 'w') as fout:
            fout.write('1.0 0:0.5 2:1.2\n')
            fout.write('-2.0\n')
            fout.write('-3.0 0:0.6 1:2.4 2:1.2\n')
            fout.write('4 2:-1.2\n')

        with open(label_path, 'w') as fout:
            fout.write('1.0\n')
            fout.write('-2.0 0:0.125\n')
            fout.write('-3.0 2:1.2\n')
            fout.write('4 1:1.0 2:-1.2\n')

        data_dir = os.path.join(cwd, 'data')
        data_train = mx.io.LibSVMIter(data_libsvm=data_path, label_libsvm=label_path,
                                      data_shape=(3, ), label_shape=(3, ), batch_size=3)

        first = mx.nd.array([[0.5, 0., 1.2], [0., 0., 0.], [0.6, 2.4, 1.2]])
        second = mx.nd.array([[0., 0., -1.2], [0.5, 0., 1.2], [0., 0., 0.]])
        i = 0
        for batch in iter(data_train):
            expected = first.asnumpy() if i == 0 else second.asnumpy()
            data = data_train.getdata()
            data.check_format(True)
            assert_almost_equal(data.asnumpy(), expected)
            i += 1

    def check_libSVMIter_news_data():
        news_metadata = {
            'name': 'news20.t',
            'origin_name': 'news20.t.bz2',
            'url': "https://apache-mxnet.s3-accelerate.dualstack.amazonaws.com/gluon/dataset/news20.t.bz2",
            'feature_dim': 62060 + 1,
            'num_classes': 20,
            'num_examples': 3993,
        }
        batch_size = 33
        num_examples = news_metadata['num_examples']
        data_dir = os.path.join(os.getcwd(), 'data')
        get_bz2_data(data_dir, news_metadata['name'], news_metadata['url'],
                     news_metadata['origin_name'])
        path = os.path.join(data_dir, news_metadata['name'])
        data_train = mx.io.LibSVMIter(data_libsvm=path, data_shape=(news_metadata['feature_dim'],),
                                      batch_size=batch_size)
        for epoch in range(2):
            num_batches = 0
            for batch in data_train:
                # check the range of labels
                data = batch.data[0]
                label = batch.label[0]
                data.check_format(True)
                assert(np.sum(label.asnumpy() > 20) == 0)
                assert(np.sum(label.asnumpy() <= 0) == 0)
                num_batches += 1
            expected_num_batches = num_examples / batch_size
            assert(num_batches == int(expected_num_batches)), num_batches
            data_train.reset()

    def check_libSVMIter_exception():
        cwd = os.getcwd()
        data_path = os.path.join(cwd, 'data.t')
        label_path = os.path.join(cwd, 'label.t')
        with open(data_path, 'w') as fout:
            fout.write('1.0 0:0.5 2:1.2\n')
            fout.write('-2.0\n')
            # Below line has a neg indice. Should throw an exception
            fout.write('-3.0 -1:0.6 1:2.4 2:1.2\n')
            fout.write('4 2:-1.2\n')

        with open(label_path, 'w') as fout:
            fout.write('1.0\n')
            fout.write('-2.0 0:0.125\n')
            fout.write('-3.0 2:1.2\n')
            fout.write('4 1:1.0 2:-1.2\n')
        data_dir = os.path.join(cwd, 'data')
        data_train = mx.io.LibSVMIter(data_libsvm=data_path, label_libsvm=label_path,
                                      data_shape=(3, ), label_shape=(3, ), batch_size=3)
        for batch in iter(data_train):
            data_train.get_data().asnumpy()

    check_libSVMIter_synthetic()
    check_libSVMIter_news_data()
    assertRaises(MXNetError, check_libSVMIter_exception)


def test_DataBatch():
    from nose.tools import ok_
    from mxnet.io import DataBatch
    import re
    batch = DataBatch(data=[mx.nd.ones((2, 3))])
    ok_(re.match(
        'DataBatch: data shapes: \[\(2L?, 3L?\)\] label shapes: None', str(batch)))
    batch = DataBatch(data=[mx.nd.ones((2, 3)), mx.nd.ones(
        (7, 8))], label=[mx.nd.ones((4, 5))])
    ok_(re.match(
        'DataBatch: data shapes: \[\(2L?, 3L?\), \(7L?, 8L?\)\] label shapes: \[\(4L?, 5L?\)\]', str(batch)))


def test_CSVIter():
    def check_CSVIter_synthetic(dtype='float32'):
        cwd = os.getcwd()
        data_path = os.path.join(cwd, 'data.t')
        label_path = os.path.join(cwd, 'label.t')
        entry_str = '1'
        if dtype is 'int32':
            entry_str = '200000001'
        if dtype is 'int64':
            entry_str = '2147483648'
        with open(data_path, 'w') as fout:
            for i in range(1000):
                fout.write(','.join([entry_str for _ in range(8*8)]) + '\n')
        with open(label_path, 'w') as fout:
            for i in range(1000):
                fout.write('0\n')

        data_train = mx.io.CSVIter(data_csv=data_path, data_shape=(8, 8),
                                   label_csv=label_path, batch_size=100, dtype=dtype)
        expected = mx.nd.ones((100, 8, 8), dtype=dtype) * int(entry_str)
        for batch in iter(data_train):
            data_batch = data_train.getdata()
            assert_almost_equal(data_batch.asnumpy(), expected.asnumpy())
            assert data_batch.asnumpy().dtype == expected.asnumpy().dtype

    for dtype in ['int32', 'int64', 'float32']:
        check_CSVIter_synthetic(dtype=dtype)

def test_ImageRecordIter_seed_augmentation():
    get_cifar10()
    seed_aug = 3

    def assert_dataiter_items_equals(dataiter1, dataiter2):
        """
        Asserts that two data iterators have the same numbner of batches,
        that the batches have the same number of items, and that the items
        are the equal.
        """
        for batch1, batch2 in zip_longest(dataiter1, dataiter2):
            
            # ensure iterators contain the same number of batches
            # zip_longest will return None if on of the iterators have run out of batches
            assert batch1 and batch2, 'The iterators do not contain the same number of batches'

            # ensure batches are of same length
            assert len(batch1.data) == len(batch2.data), 'The returned batches are not of the same length'

            # ensure batch data is the same
            for i in range(0, len(batch1.data)):
                data1 = batch1.data[i].asnumpy().astype(np.uint8)
                data2 = batch2.data[i].asnumpy().astype(np.uint8)
                assert(np.array_equal(data1, data2))

    def assert_dataiter_items_not_equals(dataiter1, dataiter2):
        """
        Asserts that two data iterators have the same numbner of batches,
        that the batches have the same number of items, and that the items
        are the _not_ equal.
        """
        for batch1, batch2 in zip_longest(dataiter1, dataiter2):

            # ensure iterators are of same length
            # zip_longest will return None if on of the iterators have run out of batches
            assert batch1 and batch2, 'The iterators do not contain the same number of batches'

            # ensure batches are of same length
            assert len(batch1.data) == len(batch2.data), 'The returned batches are not of the same length'

            # ensure batch data is the same
            for i in range(0, len(batch1.data)):
                data1 = batch1.data[i].asnumpy().astype(np.uint8)
                data2 = batch2.data[i].asnumpy().astype(np.uint8)
                if not np.array_equal(data1, data2):
                    return
        assert False, 'Expected data iterators to be different, but they are the same'

    # check whether to get constant images after fixing seed_aug
    dataiter1 = mx.io.ImageRecordIter(
        path_imgrec="data/cifar/train.rec",
        mean_img="data/cifar/cifar10_mean.bin",
        shuffle=False,
        data_shape=(3, 28, 28),
        batch_size=3,
        rand_crop=True,
        rand_mirror=True,
        max_random_scale=1.3,
        max_random_illumination=3,
        max_rotate_angle=10,
        random_l=50,
        random_s=40,
        random_h=10,
        max_shear_ratio=2,
        seed_aug=seed_aug)

    dataiter2 = mx.io.ImageRecordIter(
        path_imgrec="data/cifar/train.rec",
        mean_img="data/cifar/cifar10_mean.bin",
        shuffle=False,
        data_shape=(3, 28, 28),
        batch_size=3,
        rand_crop=True,
        rand_mirror=True,
        max_random_scale=1.3,
        max_random_illumination=3,
        max_rotate_angle=10,
        random_l=50,
        random_s=40,
        random_h=10,
        max_shear_ratio=2,
        seed_aug=seed_aug)
    
    assert_dataiter_items_equals(dataiter1, dataiter2)

    # check whether to get different images after change seed_aug
    dataiter1.reset()
    dataiter2 = mx.io.ImageRecordIter(
        path_imgrec="data/cifar/train.rec",
        mean_img="data/cifar/cifar10_mean.bin",
        shuffle=False,
        data_shape=(3, 28, 28),
        batch_size=3,
        rand_crop=True,
        rand_mirror=True,
        max_random_scale=1.3,
        max_random_illumination=3,
        max_rotate_angle=10,
        random_l=50,
        random_s=40,
        random_h=10,
        max_shear_ratio=2,
        seed_aug=seed_aug+1)

    assert_dataiter_items_not_equals(dataiter1, dataiter2)

    # check whether seed_aug changes the iterator behavior
    dataiter1 = mx.io.ImageRecordIter(
        path_imgrec="data/cifar/train.rec",
        mean_img="data/cifar/cifar10_mean.bin",
        shuffle=False,
        data_shape=(3, 28, 28),
        batch_size=3,
        seed_aug=seed_aug)

    dataiter2 = mx.io.ImageRecordIter(
        path_imgrec="data/cifar/train.rec",
        mean_img="data/cifar/cifar10_mean.bin",
        shuffle=False,
        data_shape=(3, 28, 28),
        batch_size=3,
        seed_aug=seed_aug)
    
    assert_dataiter_items_equals(dataiter1, dataiter2)

if __name__ == "__main__":
    test_NDArrayIter()
    if h5py:
        test_NDArrayIter_h5py()
    test_MNISTIter()
    test_Cifar10Rec()
    test_LibSVMIter()
    test_NDArrayIter_csr()
    test_CSVIter()
    test_ImageRecordIter_seed_augmentation()
    test_image_iter_exception()
