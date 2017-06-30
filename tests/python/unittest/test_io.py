# pylint: skip-file
import mxnet as mx
from mxnet.test_utils import *
import numpy as np
import os, gzip
import pickle as pickle
import time
import sys
from common import get_data

def test_MNISTIter():
    # prepare data
    get_data.GetMNIST_ubyte()

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
    # skip-this test for saving time
    return
    get_data.GetCifar10()
    dataiter = mx.io.ImageRecordIter(
            path_imgrec="data/cifar/train.rec",
            mean_img="data/cifar/cifar10_mean.bin",
            rand_crop=False,
            and_mirror=False,
            shuffle=False,
            data_shape=(3,28,28),
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

def test_NDArrayIter():
    datas = np.ones([1000, 2, 2])
    labels = np.ones([1000, 1])
    for i in range(1000):
        datas[i] = i / 100
        labels[i] = i / 100
    dataiter = mx.io.NDArrayIter(datas, labels, 128, True, last_batch_handle='pad')
    batchidx = 0
    for batch in dataiter:
        batchidx += 1
    assert(batchidx == 8)
    dataiter = mx.io.NDArrayIter(datas, labels, 128, False, last_batch_handle='pad')
    batchidx = 0
    labelcount = [0 for i in range(10)]
    for batch in dataiter:
        label = batch.label[0].asnumpy().flatten()
        assert((batch.data[0].asnumpy()[:,0,0] == label).all())
        for i in range(label.shape[0]):
            labelcount[int(label[i])] += 1

    for i in range(10):
        if i == 0:
            assert(labelcount[i] == 124)
        else:
            assert(labelcount[i] == 100)

def test_NDArrayIter_csr():
    import scipy.sparse as sp
    # creating toy data
    num_rows = rnd.randint(5, 15)
    num_cols = rnd.randint(1, 20)
    batch_size = rnd.randint(1, num_rows)
    shape = (num_rows, num_cols)
    csr, _ = rand_sparse_ndarray(shape, 'csr')
    dns = csr.asnumpy()

    # make iterators
    csr_iter = iter(mx.io.NDArrayIter(csr, csr, batch_size))
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
    #TODO(haibin) automatic the test instead of hard coded test
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

    data_dir = os.path.join(os.getcwd(), 'data')
    f = (data_path, label_path, (3,), (3,), 3)
    data_train = mx.io.LibSVMIter(data_libsvm=f[0],
                                  label_libsvm=f[1],
                                  data_shape=f[2],
                                  label_shape=f[3],
                                  batch_size=f[4])

    first = mx.nd.array([[ 0.5, 0., 1.2], [ 0., 0., 0.], [ 0.6, 2.4, 1.2]])
    second = mx.nd.array([[ 0., 0., -1.2], [ 0.5, 0., 1.2], [ 0., 0., 0.]])
    i = 0
    for batch in iter(data_train):
        expected = first.asnumpy() if i == 0 else second.asnumpy()
        assert_almost_equal(data_train.getdata().asnumpy(), expected)
        i += 1
'''
def test_LibSVMIter():
    kdda = os.path.join(os.getcwd(), 'news20.t')
    data_train = mx.io.LibSVMIter(data_libsvm=kdda,
                                  #label_libsvm=kdda,
                                  data_shape=(62060, ),
                                  #label_shape=f[3],
                                  batch_size=512)
    it = iter(data_train)
    for batch in it:
        print(batch.data[0])
        #break
'''

if __name__ == "__main__":
    test_NDArrayIter()
    test_MNISTIter()
    test_Cifar10Rec()
    test_LibSVMIter()
    test_NDArrayIter_csr()
