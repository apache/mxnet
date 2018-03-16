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

import shutil
from mxnet.contrib.summary import SummaryWriter
from mxnet.contrib.summary.utils import _make_metadata_tsv, make_image_grid, _make_sprite_image
from mxnet.contrib.summary.utils import _add_embedding_config, _save_embedding_tsv
from mxnet.test_utils import *
from common import with_seed

# DO NOT CHANGE THESE NAMES AS THEY FOLLOW THE DEFINITIONS IN TENSORBOARD
_LOGDIR = './logs_for_tensorboard'
_METADATA_TSV = 'metadata.tsv'
_SPRITE_PNG = 'sprite.png'
_PROJECTOR_CONFIG_PBTXT = 'projector_config.pbtxt'
_TENSORS_TSV = 'tensors.tsv'
_EVENT_FILE_PREFIX = 'events.out.tfevents'
_PLUGINS = 'plugins'
_TENSORBOARD_TEXT = 'tensorboard_text'
_TENSORS_JSON = 'tensors.json'


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


def safe_remove_dir(dir_path):
    if dir_empty(dir_path):
        try:
            shutil.rmtree(dir_path)
        except:
            raise OSError('failed to remove dir {}'.format(dir_path))


def safe_remove_logdir():
    safe_remove_dir(_LOGDIR)


def file_exists(file_path):
    return os.path.exists(file_path) and os.path.isfile(file_path)


def dir_exists(dir_path):
    return os.path.exists(dir_path) and os.path.isdir(dir_path)


def file_exists_with_prefix(file_path, prefix=None):
    if prefix is None:
        return file_exists(file_path)
    filename = os.path.basename(file_path)
    if filename.startswith(prefix):
        return True
    return False


def dir_empty(dir_path):
    for _, dirnames, files in os.walk(dir_path):
        if len(dirnames) != 0 or len(files) != 0:
            return False
    return True


def logdir_empty():
    return dir_empty(_LOGDIR)


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


def check_event_file_and_remove_logdir():
    """Check whether the event file exists and then remove the logdir."""
    files = os.listdir(_LOGDIR)
    assert len(files) == 1
    file_path = os.path.join(_LOGDIR, files[0])
    assert file_exists_with_prefix(file_path, _EVENT_FILE_PREFIX)
    safe_remove_file(file_path)
    safe_remove_logdir()


@with_seed()
def test_add_scalar():
    sw = SummaryWriter(logdir=_LOGDIR)
    sw.add_scalar(tag='test_add_scalar', value=np.random.uniform(), global_step=0)
    sw.close()
    check_event_file_and_remove_logdir()


@with_seed()
def test_add_histogram():
    shape = rand_shape_nd(4)
    sw = SummaryWriter(logdir=_LOGDIR)
    sw.add_histogram(tag='test_add_histogram', values=mx.nd.random.normal(shape=shape), global_step=0, bins=100)
    sw.close()
    check_event_file_and_remove_logdir()


@with_seed()
def test_add_image():
    shape = list(rand_shape_nd(4))
    shape[1] = 3
    shape = tuple(shape)
    sw = SummaryWriter(logdir=_LOGDIR)
    sw.add_image(tag='test_add_image', image=mx.nd.random.normal(shape=shape), global_step=0)
    sw.close()
    check_event_file_and_remove_logdir()


@with_seed()
def test_add_audio():
    shape = (100,)
    data = mx.nd.random.uniform(-1, 1, shape=shape)
    sw = SummaryWriter(logdir=_LOGDIR)
    sw.add_audio(tag='test_add_audio', audio=data)
    sw.close()
    check_event_file_and_remove_logdir()


def check_and_remove_logdir_for_text():
    """1. verify that tensors.json exists under _LOGDIR/plugins/tensorboard_text.
    2. verify that the event files exists and remove it."""
    # step 1
    plugins_path = os.path.join(_LOGDIR, _PLUGINS)
    tensorboard_text_path = os.path.join(plugins_path, _TENSORBOARD_TEXT)
    file_path = os.path.join(tensorboard_text_path, _TENSORS_JSON)
    assert file_exists(file_path)
    safe_remove_file(file_path)
    safe_remove_dir(tensorboard_text_path)
    safe_remove_dir(plugins_path)
    # step 2
    event_files = os.listdir(_LOGDIR)
    assert len(event_files) == 1
    event_file_path = os.path.join(_LOGDIR, event_files[0])
    assert file_exists_with_prefix(event_file_path, _EVENT_FILE_PREFIX)
    safe_remove_file(event_file_path)
    # remove logdir
    safe_remove_logdir()


def test_add_text():
    # this will generate an event file under _LOGDIR and
    # a json file called tensors.json under _LOGDIR/plugins/tensorboard_text/tensors.json
    sw = SummaryWriter(logdir=_LOGDIR)
    sw.add_text(tag='test_add_text', text='Hello MXNet!')
    sw.close()
    check_and_remove_logdir_for_text()


def check_and_remove_for_embedding(global_step):
    """1. verify projector_config.pbtxt exists under _LOGDIR.
    2. verify folder str(global_step).zfill(5) exists under _LOGDIR.
    3. verify metadata.tsv exists under _LOGDIR/str(global_step).zfill(5).
    4. verify sprinte.png exists under _LOGDIR/str(global_step).zfill(5).
    5. verify tensors.tsv exists under _LOGDIR/str(global_step).zfill(5).
    6. remove all of them and _LOGDIR."""
    # step 1
    projector_file_path = os.path.join(_LOGDIR, _PROJECTOR_CONFIG_PBTXT)
    assert file_exists(projector_file_path)
    global_step_dir_path = os.path.join(_LOGDIR, str(global_step).zfill(5))
    assert dir_exists(global_step_dir_path)
    metadata_tsv_path = os.path.join(global_step_dir_path, _METADATA_TSV)
    assert file_exists(metadata_tsv_path)
    sprint_png_path = os.path.join(global_step_dir_path, _SPRITE_PNG)
    assert file_exists(sprint_png_path)
    tensors_tsv_path = os.path.join(global_step_dir_path, _TENSORS_TSV)
    assert file_exists(tensors_tsv_path)

    safe_remove_file(projector_file_path)
    safe_remove_file(metadata_tsv_path)
    safe_remove_file(sprint_png_path)
    safe_remove_file(tensors_tsv_path)
    safe_remove_dir(global_step_dir_path)
    safe_remove_logdir()


@with_seed()
def test_add_embedding():
    batch_size = 10
    embedding = mx.nd.uniform(shape=(batch_size, 20))
    labels = mx.nd.uniform(low=1, high=2, shape=(batch_size,)).astype('int32')
    images = mx.nd.uniform(shape=(batch_size, 3, 10, 10))
    global_step = np.random.randint(low=0, high=999999)
    with SummaryWriter(logdir=_LOGDIR) as sw:
        sw.add_embedding(tag='test_add_embedding', embedding=embedding, labels=labels,
                         images=images, global_step=global_step)
    check_and_remove_for_embedding(global_step)


@with_seed()
def test_add_pr_curve():
    shape = (100,)
    predictions = mx.nd.uniform(low=0.0, high=1.0, shape=shape)
    labels = mx.nd.uniform(low=0, high=2, shape=shape).astype('int32')
    num_threshodls = 100
    with SummaryWriter(_LOGDIR) as sw:
        sw.add_pr_curve(tag='test_add_pr_curve', labels=labels, predictions=predictions, num_thresholds=num_threshodls)
    check_event_file_and_remove_logdir()


if __name__ == '__main__':
    import nose
    nose.runmodule()
