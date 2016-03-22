#################################################################################
# This is a functional test for the currently internal-testing feature of
# "mirroring" some nodes in the computation graph to reduce memory consumption
# with computation. Briefly, the feed-forward intermediate results for all
# operators typically saved if they are needed for backward computation. However,
# for some operators, computation is cheap, so we can discard the intermediate
# results and repeat that forward computation during when needed. Detailed
# documentation could be expected when this feature is mature.
#
# When mirroring is turned on and set properly, we could expect smaller memory
# consumption with slightly slower computation speed (due to extra forward 
# steps). We are not including a sample running log here, as this test case
# is only a functionality test. The using of pycuda GPU memory query is also
# not very good way of measuring the memory usage here.
#################################################################################
import find_mxnet
import mxnet as mx
import argparse
import os, sys
import train_model

parser = argparse.ArgumentParser(description='train an image classifer on cifar10')
parser.add_argument('--network', type=str, default='inception-bn-28-small',
                    help = 'the cnn to use')
parser.add_argument('--data-dir', type=str, default='cifar10/',
                    help='the input data directory')
parser.add_argument('--gpus', type=str, default='0',
                    help='the gpus will be used, e.g "0,1,2,3"')
parser.add_argument('--num-examples', type=int, default=60000,
                    help='the number of training examples')
parser.add_argument('--batch-size', type=int, default=128,
                    help='the batch size')
parser.add_argument('--lr', type=float, default=.05,
                    help='the initial learning rate')
parser.add_argument('--lr-factor', type=float, default=1,
                    help='times the lr with a factor for every lr-factor-epoch epoch')
parser.add_argument('--lr-factor-epoch', type=float, default=1,
                    help='the number of epoch to factor the lr, could be .5')
parser.add_argument('--model-prefix', type=str,
                    help='the prefix of the model to load/save')
parser.add_argument('--save-model-prefix', type=str,
                    help='the prefix of the model to save')
parser.add_argument('--num-epochs', type=int, default=1,
                    help='the number of training epochs')
parser.add_argument('--load-epoch', type=int,
                    help="load the model on an epoch using the model-prefix")
parser.add_argument('--kv-store', type=str, default='local',
                    help='the kvstore type')
args = parser.parse_args()

# download data if necessary
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

# network
import importlib
net_gen = importlib.import_module('symbol_' + args.network)
net = net_gen.get_symbol(10)
net_mirrored = net_gen.get_symbol(10, force_mirroring=True)

# data
def get_iterator(args, kv):
    data_shape = (3, 28, 28)
    if '://' not in args.data_dir:
        _download(args.data_dir)

    train = mx.io.ImageRecordIter(
        path_imgrec = args.data_dir + "train.rec",
        mean_img    = args.data_dir + "mean.bin",
        data_shape  = data_shape,
        batch_size  = args.batch_size,
        rand_crop   = True,
        rand_mirror = True,
        num_parts   = kv.num_workers,
        part_index  = kv.rank)

    val = mx.io.ImageRecordIter(
        path_imgrec = args.data_dir + "test.rec",
        mean_img    = args.data_dir + "mean.bin",
        rand_crop   = False,
        rand_mirror = False,
        data_shape  = data_shape,
        batch_size  = args.batch_size,
        num_parts   = kv.num_workers,
        part_index  = kv.rank)

    return (train, val)

# call back to report memory consumption
import pycuda.autoinit
import pycuda.driver as cuda
import logging
def report_gpu_memory(every_n_batch=50):
    def __callback(param):
        if param.nbatch % every_n_batch == 0:
            (free, total) = cuda.mem_get_info()
            logging.info('        GPU Memory: %.2f%%' % (100.0*free / total))
    return __callback

################################################################################
print("*" * 80)
print("  WITHOUT mirroring")
print("*" * 80)

# train
train_model.fit(args, net, get_iterator, batch_end_callback=report_gpu_memory())

################################################################################
print("*" * 80)
print("  WITH mirroring via attributes")
print("*" * 80)

# train
train_model.fit(args, net_mirrored, get_iterator, batch_end_callback=report_gpu_memory())

################################################################################
import os
os.environ['MXNET_BACKWARD_DO_MIRROR'] = '1'
print("*" * 80)
print("  WITH mirroring via environment variable")
print("*" * 80)

# train
train_model.fit(args, net, get_iterator, batch_end_callback=report_gpu_memory())
