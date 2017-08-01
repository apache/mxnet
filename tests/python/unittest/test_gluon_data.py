import os
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


def prepare_record():
    if not os.path.isdir("data"):
        os.makedirs('data')
    if not os.path.isdir("data/test_images"):
        os.system("wget http://data.mxnet.io/data/test_images.tar.gz -O data/test_images.tar.gz")
        os.system("tar -xf data/test_images.tar.gz -C data")
    imgs = os.listdir('data/test_images')
    record = mx.recordio.MXIndexedRecordIO('data/test.idx', 'data/test.rec', 'w')
    for i, img in enumerate(imgs):
        str_img = open('data/test_images/'+img, 'rb').read()
        s = mx.recordio.pack((0, i, i, 0), str_img)
        record.write_idx(i, s)
    return 'data/test.rec'


def test_recordimage_dataset():
    recfile = prepare_record()
    dataset = gluon.data.ImageRecordDataset(recfile)
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

if __name__ == '__main__':
    import nose
    nose.runmodule()
