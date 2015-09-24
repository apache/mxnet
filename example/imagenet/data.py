# pylint: skip-file
""" data iterator for imagnet"""
import sys
sys.path.insert(0, "../../python/")
import mxnet as mx

def ilsvrc12_iterator(batch_size, input_shape):
    """return train and val iterators for imagenet"""
    train_dataiter = mx.io.ImageRecordIter(
        path_imgrec        = "data/ilsvrc12/train.rec",
        mean_img           = "data/ilsvrc12/mean.bin",
        rand_crop          = True,
        rand_mirror        = True,
        prefetch_buffer    = 4,
        preprocess_threads = 4,
        data_shape         = input_shape,
        batch_size         = batch_size)
    val_dataiter = mx.io.ImageRecordIter(
        path_imgrec        = "data/ilsvrc12/val.rec",
        mean_img           = "data/ilsvrc12/mean.bin",
        rand_crop          = False,
        rand_mirror        = False,
        prefetch_buffer    = 4,
        preprocess_threads = 4,
        data_shape         = input_shape,
        batch_size         = batch_size)

    return (train_dataiter, val_dataiter)
