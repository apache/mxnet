import find_mxnet
import mxnet as mx
import logging
import argparse
import os, sys
logging.basicConfig(level=logging.DEBUG)

parser = argparse.ArgumentParser(description='train an image classifer on cifar10')
parser.add_argument('--network', type=str, default='inception_bn_28',
                    help = 'the cnn to use')
parser.add_argument('--data-dir', type=str, default='cifar10/',
                    help='the input data directory')
parser.add_argument('--lr', type=float, default=.05,
                    help='the initial learning rate')
parser.add_argument('--num-epochs', type=int, default=10,
                    help='the number of training epochs')
parser.add_argument('--batch-size', type=int, default=128,
                    help='the batch size')
parser.add_argument('--gpus', type=str, default='0',
                    help='the gpus will be used, e.g "0,1,2,3"')
args = parser.parse_args()

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
                 data_dir,
                 input_shape = (3,28,28),
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
        batch_size  = batch_size,
        num_parts   = num_parts,
        part_index  = part_index)
    return (train, val)

train, val = get_iterator(args.batch_size, args.data_dir)

import importlib
net = importlib.import_module(args.network)

model = mx.model.FeedForward.create(
    ctx                = [mx.gpu(int(i)) for i in args.gpus.split(',')],
    symbol             = net.get_symbol(10),
    num_epoch          = args.num_epochs,
    learning_rate      = args.lr,
    momentum           = 0.9,
    wd                 = 0.00001,
    X                  = train,
    eval_data          = val,
    batch_end_callback = mx.callback.Speedometer(args.batch_size))
