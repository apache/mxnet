import find_mxnet
import mxnet as mx
import logging
import argparse

# don't use -n and -s, which are resevered for the distributed training
parser = argparse.ArgumentParser(description='train an image classifer on imagenet')
parser.add_argument('--network', type=str, default='alexnet',
                    choices = ['alexnet', 'vgg', 'googlenet', 'inception-bn'],
                    help = 'the cnn to use')
parser.add_argument('--data-dir', type=str, required=True,
                    help='the input data directory')
parser.add_argument('--model-prefix', type=str,
                    help='the prefix of the model')
parser.add_argument('--lr', type=float, default=.05,
                    help='the initial learning rate')
parser.add_argument('--num-epochs', type=int, default=20,
                    help='the number of training epochs')
parser.add_argument('--load-epoch', type=int,
                    help="load the model on an epoch using the model-prefix")
parser.add_argument('--batch-size', type=int, default=32,
                    help='the batch size')
parser.add_argument('--gpus', type=str, default='0',
                    help='the gpus will be used, e.g "0,1,2,3"')
parser.add_argument('--kv-type', type=str, default='local',
                    help='the kvstore type')
parser.add_argument('--num-examples', type=int, default=1281167,
                    help='the number of training examples')
parser.add_argument('--num-classes', type=int, default=1000,
                    help='the number of classes')
args = parser.parse_args()

# network
import importlib
net = importlib.import_module(args.network)

# kvstore
kv = mx.kvstore.create(args.kv_type)
head = '%(asctime)-15s Node[' + str(kv.rank) + '] %(message)s'
logging.basicConfig(level=logging.DEBUG, format=head)

# data
data_shape = (3, 224, 224)
print args.data_dir
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
    path_imgrec = args.data_dir + "val.rec",
    mean_img    = args.data_dir + "mean.bin",
    rand_crop   = False,
    rand_mirror = False,
    data_shape  = data_shape,
    batch_size  = args.batch_size,
    num_parts   = kv.num_workers,
    part_index  = kv.rank)

# load / save model?
model_prefix = args.model_prefix
checkpoint = None
if model_prefix is not None:
    model_prefix += "-%d" % (kv.rank)
    checkpoint = mx.callback.do_checkpoint(model_prefix)

load_model = {}
if args.load_epoch is not None:
    assert model_prefix is not None
    tmp = mx.model.FeedForward.load(model_prefix, args.load_epoch)
    load_model = {'arg_params' : tmp.arg_params,
                  'aux_params' : tmp.aux_params,
                  'begin_epoch' : args.load_epoch}

# train
model = mx.model.FeedForward(
    ctx                = [mx.gpu(int(i)) for i in args.gpus.split(',')],
    symbol             = net.get_symbol(args.num_classes),
    num_epoch          = args.num_epochs,
    learning_rate      = args.lr,
    momentum           = 0.9,
    wd                 = 0.00001,
    **load_model)

model.fit(
    X                  = train,
    eval_data          = val,
    kvstore            = kv,
    batch_end_callback = mx.callback.Speedometer(args.batch_size, 50),
    epoch_end_callback = checkpoint)
