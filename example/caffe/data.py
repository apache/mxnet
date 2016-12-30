import sys
import os
# code to automatically download dataset
curr_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
sys.path.append(os.path.join(curr_path, "../../tests/python/common"))
import get_data
import mxnet as mx

def get_iterator(data_shape, use_caffe_data):
    def get_iterator_impl_mnist(args, kv):
        """return train and val iterators for mnist"""
        # download data
        get_data.GetMNIST_ubyte()
        flat = False if len(data_shape) != 1 else True

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

    def get_iterator_impl_caffe(args, kv):
        flat = False if len(data_shape) != 1 else True
        train = mx.io.CaffeDataIter(
            prototxt =
            'layer { \
                name: "mnist" \
                type: "Data" \
                top: "data" \
                top: "label" \
                include { \
                    phase: TRAIN \
                } \
                transform_param { \
                    scale: 0.00390625 \
                } \
                data_param { \
                    source: "mnist_train_lmdb" \
                    batch_size: 64 \
                    backend: LMDB \
                } \
            }',
            flat           = flat,
            num_examples   = 60000
            # float32 is the default, so left out here in order to illustrate
        )

        val = mx.io.CaffeDataIter(
            prototxt =
            'layer { \
                name: "mnist" \
                type: "Data" \
                top: "data" \
                top: "label" \
                include { \
                    phase: TEST \
                } \
                transform_param { \
                    scale: 0.00390625 \
                } \
                data_param { \
                    source: "mnist_test_lmdb" \
                    batch_size: 100 \
                    backend: LMDB \
                } \
            }',
            flat           = flat,
            num_examples   = 10000,
            dtype          = "float32" # float32 is the default
        )

        return train, val

    if use_caffe_data:
        return get_iterator_impl_caffe
    else:
        return get_iterator_impl_mnist
