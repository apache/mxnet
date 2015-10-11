# pylint: skip-file
""" data iterator for multi-node.

all iterators are disabled randomness

must create kv before
"""
import sys
sys.path.insert(0, "../common/")
sys.path.insert(0, "../../python/")
import mxnet as mx
import get_data

def mnist(batch_size, input_shape, num_parts=1, part_index=0):
    """return mnist iters"""
    get_data.GetMNIST_ubyte()
    flat = len(input_shape)==1
    train = mx.io.MNISTIter(
        image      = "data/train-images-idx3-ubyte",
        label      = "data/train-labels-idx1-ubyte",
        data_shape = input_shape,
        batch_size = batch_size,
        num_parts  = num_parts,
        part_index = part_index,
        shuffle    = False,
        flat       = flat,
        silent     = False)
    val = mx.io.MNISTIter(
        image      = "data/t10k-images-idx3-ubyte",
        label      = "data/t10k-labels-idx1-ubyte",
        data_shape = input_shape,
        batch_size = batch_size,
        shuffle    = False,
        flat       = flat,
        silent     = False)
    return (train, val)
