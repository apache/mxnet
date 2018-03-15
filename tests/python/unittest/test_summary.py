import os
import shutil
import numpy as np
import mxnet as mx
from mxnet.contrib.summary import SummaryWriter
from mxnet.contrib.summary.utils import _make_metadata_tsv
from mxnet.test_utils import rand_shape_nd, rand_ndarray, same
from common import with_seed

_LOGDIR = './logs_for_tensorboard'


def make_logdir():
    if not os.path.exists(_LOGDIR):
        try:
            os.mkdir(_LOGDIR)
        except:
            raise OSError('failed to make dir at {}'.format(_LOGDIR))


def remove_file(file_path):
    if file_exists(file_path):
        try:
            os.remove(file_path)
        except:
            raise OSError('failed to remove file at {}'.format(file_path))


def safe_remove_logdir():
    if logdir_empty():
        try:
            shutil.rmtree(_LOGDIR)
        except:
            raise OSError('failed to remove logdir at {}, please make sure it is empty before deletion'.format(_LOGDIR))


def file_exists(file_path):
    return os.path.exists(file_path) and os.path.isfile(file_path)


def logdir_empty():
    for dirpath, dirnames, files in os.walk(_LOGDIR):
        if files:
            return False
    return True


@with_seed()
def test_make_metadata_tsv():
    shape = rand_shape_nd(num_dim=4, dim=10)
    data = rand_ndarray(shape=shape, stype='default')
    _make_metadata_tsv(data, _LOGDIR)
    file_path = os.path.join(_LOGDIR, 'metadata.tsv')
    data_loaded = np.loadtxt(file_path, dtype=data.dtype)
    assert same(data.asnumpy(), data_loaded.reshape(data.shape))
    remove_file(file_path)


if __name__ == '__main__':
    make_logdir()
    test_make_metadata_tsv()
    safe_remove_logdir()
