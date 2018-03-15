import os
import shutil
import numpy as np
import mxnet as mx
from mxnet.contrib.summary import SummaryWriter
from mxnet.contrib.summary.utils import _make_metadata_tsv, make_image_grid, _make_sprite_image
from mxnet.contrib.summary.utils import _add_embedding_config, _save_embedding_tsv
from mxnet.test_utils import *
from common import with_seed

# DO NOT CHANGE THESE
_LOGDIR = './logs_for_tensorboard'
_METADATA_FILENAME = 'metadata.tsv'
_SPRITE_PNG = 'sprite.png'
_PROJECTOR_CONFIG_PBTXT = 'projector_config.pbtxt'
_TENSORS_TSV = 'tensors.tsv'
_EVENT_FILE_PREFIX = 'events.out.tfevents'


def make_logdir():
    if not os.path.exists(_LOGDIR):
        try:
            os.mkdir(_LOGDIR)
        except:
            raise OSError('failed to make dir at {}'.format(_LOGDIR))


def safe_remove_file(file_path):
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


def file_exists_with_substr(file_path, substr=None):
    if substr is None:
        return file_exists(file_path)
    filename = os.path.basename(file_path)
    if filename.find(substr) != -1:
        return True
    return False


def file_exists_with_prefix(file_path, prefix=None):
    if prefix is None:
        return file_exists(file_path)
    filename = os.path.basename(file_path)
    if filename.startswith(prefix):
        return True
    return False


def logdir_empty():
    for dirpath, dirnames, files in os.walk(_LOGDIR):
        if files:
            return False
    return True


@with_seed()
def test_make_metadata_tsv():
    make_logdir()
    shape = rand_shape_nd(num_dim=4, dim=10)
    data = rand_ndarray(shape=shape, stype='default')
    _make_metadata_tsv(data, _LOGDIR)
    file_path = os.path.join(_LOGDIR, 'metadata.tsv')
    data_loaded = np.loadtxt(file_path, dtype=data.dtype)
    assert same(data.asnumpy(), data_loaded.reshape(data.shape))
    safe_remove_file(file_path)
    safe_remove_logdir()


@with_seed()
def test_make_image_grid():
    def test_2d_input():
        shape = rand_shape_2d()
        data = rand_ndarray(shape, 'default')
        grid = make_image_grid(data)
        assert grid.ndim == 3
        assert grid.shape[0] == 3
        assert grid.shape[1:] == data.shape
        assert same(grid[0].asnumpy(), grid[1].asnumpy())
        assert same(grid[0].asnumpy(), grid[2].asnumpy())
        assert same(grid[0].asnumpy(), data.asnumpy())

    def test_3d_single_channel_input():
        shape = rand_shape_3d(dim0=1)
        data = rand_ndarray(shape, 'default')
        assert data.shape[0] == 1  # single channel
        grid = make_image_grid(data)
        assert grid.ndim == 3
        assert grid.shape[0] == 3
        assert same(grid[0].asnumpy(), grid[1].asnumpy())
        assert same(grid[0].asnumpy(), grid[2].asnumpy())
        assert same(grid[0:1].asnumpy(), data.asnumpy())

    def test_3d_three_channel_input():
        shape = rand_shape_3d()
        shape = (3,) + shape[1:]
        data = rand_ndarray(shape, 'default')
        grid = make_image_grid(data)
        assert grid.ndim == 3
        assert grid.shape[0] == 3
        assert same(grid.asnumpy(), data.asnumpy())

    def test_4d_single_batch_single_channel_input():
        shape = list(rand_shape_nd(4))
        shape[0] = 1
        shape[1] = 1
        shape = tuple(shape)
        data = rand_ndarray(shape, 'default')
        grid = make_image_grid(data)
        assert grid.ndim == 3
        assert grid.shape[0] == 3
        assert same(grid[0].asnumpy(), grid[1].asnumpy())
        assert same(grid[0].asnumpy(), grid[2].asnumpy())
        assert same(grid[0].reshape(data.shape).asnumpy(), data.asnumpy())

    def test_4d_multiple_batch_input():
        shape_list = list(rand_shape_nd(4))
        shape_list[0] = 10
        num_channels = [1, 3]
        for c in num_channels:
            shape_list[1] = c
            shape = tuple(shape_list)
            data = rand_ndarray(shape, 'default')
            grid = make_image_grid(data)
            assert grid.ndim == 3
            assert grid.shape[0] == 3

    test_2d_input()
    test_3d_single_channel_input()
    test_3d_three_channel_input()
    test_4d_single_batch_single_channel_input()
    test_4d_multiple_batch_input()


def test_make_sprite_image():
    dtypes = [np.uint8, np.float32, np.float64]
    ndims = [2, 3, 4]
    for dtype in dtypes:
        for ndim in ndims:
            shape_list = list(rand_shape_nd(num_dim=ndim))
            if ndim == 3:
                shape_list[0] = 3
            elif ndim == 4:
                shape_list[1] = 3
            data = rand_ndarray(tuple(shape_list), 'default', dtype=dtype)
            make_logdir()
            _make_sprite_image(data, _LOGDIR)
            file_path = os.path.join(_LOGDIR, _SPRITE_PNG)
            assert file_exists(file_path)
            safe_remove_file(file_path)
            safe_remove_logdir()


def test_add_embedding_config():
    make_logdir()
    _add_embedding_config(_LOGDIR, str(10000), True, (4, 3, 5, 5))
    file_path = os.path.join(_LOGDIR, _PROJECTOR_CONFIG_PBTXT)
    assert file_exists(file_path)
    safe_remove_file(file_path)
    safe_remove_logdir()


def test_save_ndarray_tsv():
    dtypes = [np.uint8, np.float32, np.float64]
    ndims = [2, 3, 4]
    for dtype in dtypes:
        for ndim in ndims:
            shape = rand_shape_nd(ndim)
            data = rand_ndarray(shape, 'default', dtype=dtype)
            make_logdir()
            _save_embedding_tsv(data, _LOGDIR)
            file_path = os.path.join(_LOGDIR, _TENSORS_TSV)
            safe_remove_file(file_path)
            safe_remove_logdir()


def check_and_clean_single_event_file():
    files = os.listdir(_LOGDIR)
    assert len(files) == 1
    file_path = os.path.join(_LOGDIR, files[0])
    assert file_exists_with_prefix(file_path, _EVENT_FILE_PREFIX)
    safe_remove_file(file_path)
    safe_remove_logdir()


def test_add_scalar():
    sw = SummaryWriter(logdir=_LOGDIR)
    sw.add_scalar(tag='test_add_scalar', value=10, global_step=0)
    sw.close()
    check_and_clean_single_event_file()


def test_add_histogram():
    shape = rand_shape_nd(4)
    sw = SummaryWriter(logdir=_LOGDIR)
    sw.add_histogram(tag='test_add_histogram', values=mx.nd.random.normal(shape=shape), global_step=0, bins=100)
    sw.close()
    check_and_clean_single_event_file()


def test_add_image():
    shape = list(rand_shape_nd(4))
    shape[1] = 3
    shape = tuple(shape)
    sw = SummaryWriter(logdir=_LOGDIR)
    sw.add_image(tag='test_add_image', image=mx.nd.random.normal(shape=shape), global_step=0)
    sw.close()
    check_and_clean_single_event_file()


def test_add_audio():
    shape = (100,)
    data = mx.nd.random.uniform(-1, 1, shape=shape)
    sw = SummaryWriter(logdir=_LOGDIR)
    sw.add_audio(tag='test_add_audio', audio=data)
    sw.close()
    check_and_clean_single_event_file()


if __name__ == '__main__':
    import nose
    nose.runmodule()
