import find_mxnet
import mxnet as mx
import logging
import argparse
import os
import train_model

# don't use -n and -s, which are resevered for the distributed training
parser = argparse.ArgumentParser(description='train an image classifer on imagenet')
mutually_exclusive_parser_group = parser.add_mutually_exclusive_group(required=True)
parser.add_argument('--network', type=str, default='inception-bn',
                    choices = ['alexnet', 'vgg', 'googlenet', 'inception-bn',
                               'inception-bn-full', 'inception-v3', 'resnet'],
                    help = 'the cnn to use')
mutually_exclusive_parser_group.add_argument('--data-dir', type=str,
                    help='the input data directory')
parser.add_argument('--model-prefix', type=str,
                    help='the prefix of the model to load')
parser.add_argument('--save-model-prefix', type=str,
                    help='the prefix of the model to save')
parser.add_argument('--lr', type=float, default=.01,
                    help='the initial learning rate')
parser.add_argument('--lr-factor', type=float, default=1,
                    help='times the lr with a factor for every lr-factor-epoch epoch')
parser.add_argument('--lr-factor-epoch', type=float, default=1,
                    help='the number of epoch to factor the lr, could be .5')
parser.add_argument('--clip-gradient', type=float, default=5.,
                    help='clip min/max gradient to prevent extreme value')
parser.add_argument('--num-epochs', type=int, default=20,
                    help='the number of training epochs')
parser.add_argument('--load-epoch', type=int,
                    help="load the model on an epoch using the model-prefix")
parser.add_argument('--batch-size', type=int, default=32,
                    help='the batch size')
parser.add_argument('--gpus', type=str,
                    help='gpus to be used, e.g "0,1,2,3"')
parser.add_argument('--kv-store', type=str, default='local',
                    help='the kvstore type')
parser.add_argument('--num-examples', type=int, default=1281167,
                    help='the number of training examples')
parser.add_argument('--num-classes', type=int, default=1000,
                    help='the number of classes')
parser.add_argument('--log-file', type=str,
		    help='the name of log file')
parser.add_argument('--log-dir', type=str, default="/tmp/",
                    help='directory of the log file')
parser.add_argument('--train-dataset', type=str, default="train.rec",
                    help='train dataset name')
parser.add_argument('--val-dataset', type=str, default="val.rec",
                    help="validation dataset name")
parser.add_argument('--data-shape', type=int, default=224,
                    help='set image\'s shape')
mutually_exclusive_parser_group.add_argument('--benchmark', default=False, action='store_true',
                    help='benchmark for 50 iterations using randomly generated Synthetic data')
args = parser.parse_args()

# network
import importlib
net = importlib.import_module('symbol_' + args.network).get_symbol(args.num_classes)


# data
import random
from mxnet.io import DataBatch, DataIter
import numpy as np
class SyntheticDataIter(DataIter):
    def __init__(self, num_classes, data_shape, max_iter):
        self.batch_size = data_shape[0]
        self.cur_iter = 0
        self.max_iter = max_iter
        label = np.random.randint(0, num_classes, [self.batch_size,])
        data = np.random.uniform(-1, 1, data_shape)
        self.data = mx.nd.array(data)
        self.label = mx.nd.array(label)
    def __iter__(self):
        return self
    @property
    def provide_data(self):
        return [('data',self.data.shape)]
    @property
    def provide_label(self):
        return [('softmax_label',(self.batch_size,))]
    def next(self):
        self.cur_iter += 1
        if self.cur_iter <= self.max_iter:
            return DataBatch(data=(self.data,),
                             label=(self.label,),
                             pad=0,
                             index=None,
                             provide_data=self.provide_data,
                             provide_label=self.provide_label)
        else:
            raise StopIteration
    def __next__(self):
        return self.next()
    def reset(self):
        self.cur_iter = 0

def get_sythentic_data_iter(args, kv):
    data_shape = (args.batch_size, 3, args.data_shape, args.data_shape)
    train = SyntheticDataIter(args.num_classes, data_shape, 50)
    val = SyntheticDataIter(args.num_classes, data_shape, 1)
    return (train, val)

def get_iterator(args, kv):
    data_shape = (3, args.data_shape, args.data_shape)
    train = mx.io.ImageRecordIter(
        path_imgrec = os.path.join(args.data_dir, args.train_dataset),
        mean_r      = 123.68,
        mean_g      = 116.779,
        mean_b      = 103.939,
        data_shape  = data_shape,
        batch_size  = args.batch_size,
        rand_crop   = True,
        rand_mirror = True,
        num_parts   = kv.num_workers,
        part_index  = kv.rank)

    val = mx.io.ImageRecordIter(
        path_imgrec = os.path.join(args.data_dir, args.val_dataset),
        mean_r      = 123.68,
        mean_g      = 116.779,
        mean_b      = 103.939,
        rand_crop   = False,
        rand_mirror = False,
        data_shape  = data_shape,
        batch_size  = args.batch_size,
        num_parts   = kv.num_workers,
        part_index  = kv.rank)

    return (train, val)

if args.benchmark:
    train_model.fit(args, net, get_sythentic_data_iter)
else:
    train_model.fit(args, net, get_iterator)
