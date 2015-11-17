import os
import sys
curr_path = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(curr_path, "../../../python"))
import mxnet as mx

def _download(data_dir):
    if not os.path.isdir(data_dir):
        os.system("mkdir " + data_dir)
    os.chdir(data_dir)
    if (not os.path.exists('train.rec')) or \
       (not os.path.exists('test.rec')) :
        os.system("wget http://webdocs.cs.ualberta.ca/~bx3/data/cifar10.zip")
        os.system("unzip -u cifar10.zip")
        os.system("mv cifar/* .; rm -rf cifar; rm cifar10.zip")
    os.chdir("..")

def get_iterator(batch_size,
                 input_shape = (3,28,28),
                 data_dir    = 'data/',
                 num_parts   = 1,
                 part_index  = 0):
    """return train and val iterators for cifar10"""
    if '://' not in data_dir:
        _download(data_dir)

    train = mx.io.ImageRecordIter(
        path_imgrec = data_dir + "train.rec",
        mean_img    = data_dir + "mean.bin",
        data_shape  = input_shape,
        batch_size  = batch_size,
        rand_crop   = True,
        rand_mirror = True,
        num_parts   = num_parts,
        part_index  = part_index)
    val = mx.io.ImageRecordIter(
        path_imgrec = data_dir + "test.rec",
        mean_img    = data_dir + "mean.bin",
        rand_crop   = False,
        rand_mirror = False,
        data_shape  = input_shape,
        batch_size  = batch_size)
    return (train, val)
