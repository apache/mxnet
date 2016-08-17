import sys
import os
# code to automatically download dataset
curr_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
sys.path.append(os.path.join(curr_path, "../../tests/python/common"))
import get_data
import mxnet as mx

def get_iterator(data_shape):
    def get_iterator_impl(args, kv):
        """return train and val iterators for mnist"""
        # download data
        get_data.GetMNIST_ubyte()
        flat = False if len(data_shape) == 3 else True

        train           = mx.io.MNISTIter(
            image       = "data/train-images-idx3-ubyte",
            label       = "data/train-labels-idx1-ubyte",
            input_shape = data_shape,
            batch_size  = args.batch_size,
            shuffle     = True,
            flat        = flat,
            num_parts   = kv.num_workers,
            part_index  = kv.rank)

        val = mx.io.MNISTIter(
            image       = "data/t10k-images-idx3-ubyte",
            label       = "data/t10k-labels-idx1-ubyte",
            input_shape = data_shape,
            batch_size  = args.batch_size,
            flat        = flat,
            num_parts   = kv.num_workers,
            part_index  = kv.rank)

        return (train, val)
    return get_iterator_impl

