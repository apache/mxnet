import mxnet as mx
import numpy as np
import logging
import os
from sklearn.datasets import fetch_mldata
from mxnet.quantization import *
import mxnet.ndarray as nd
import argparse

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

INFERENCE = True
no_bias = True
name = 'resnet_cifar'
data_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data/')
checkpoint_path = os.path.join('checkpoints/', name)

batch_size = 32
layer = 50
ctx = mx.gpu(0)
ignore_symbols = []

def residual_unit(data, num_filter, stride, dim_match, name, bottle_neck=True, bn_mom=0.9, workspace=256, memonger=False):
    if bottle_neck:
        # the same as https://github.com/facebook/fb.resnet.torch#notes, a bit difference with origin paper
        bn1 = mx.sym.BatchNorm(data=data, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn1')
        act1 = mx.sym.relu(data=bn1, name=name + '_relu1')
        conv1 = mx.sym.Convolution(data=act1, num_filter=int(num_filter*0.25), kernel=(1,1), stride=(1,1), pad=(0,0),
                                   no_bias=True, workspace=workspace, name=name + '_conv1')
        bn2 = mx.sym.BatchNorm(data=conv1, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn2')
        act2 = mx.sym.relu(data=bn2, name=name + '_relu2')
        conv2 = mx.sym.Convolution(data=act2, num_filter=int(num_filter*0.25), kernel=(3,3), stride=stride, pad=(1,1),
                                   no_bias=True, workspace=workspace, name=name + '_conv2')
        bn3 = mx.sym.BatchNorm(data=conv2, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn3')
        act3 = mx.sym.relu(data=bn3, name=name + '_relu3')
        conv3 = mx.sym.Convolution(data=act3, num_filter=num_filter, kernel=(1,1), stride=(1,1), pad=(0,0), no_bias=True,
                                   workspace=workspace, name=name + '_conv3')
        if dim_match:
            shortcut = data
        else:
            shortcut = mx.sym.Convolution(data=act1, num_filter=num_filter, kernel=(1,1), stride=stride, no_bias=True,
                                            workspace=workspace, name=name+'_sc')
        if memonger:
            shortcut._set_attr(mirror_stage='True')
        return conv3 + shortcut
    else:
        bn1 = mx.sym.BatchNorm(data=data, fix_gamma=False, momentum=bn_mom, eps=2e-5, name=name + '_bn1')
        act1 = mx.sym.relu(data=bn1, name=name + '_relu1')
        conv1 = mx.sym.Convolution(data=act1, num_filter=num_filter, kernel=(3,3), stride=stride, pad=(1,1),
                                      no_bias=True, workspace=workspace, name=name + '_conv1')
        bn2 = mx.sym.BatchNorm(data=conv1, fix_gamma=False, momentum=bn_mom, eps=2e-5, name=name + '_bn2')
        act2 = mx.sym.relu(data=bn2, name=name + '_relu2')
        conv2 = mx.sym.Convolution(data=act2, num_filter=num_filter, kernel=(3,3), stride=(1,1), pad=(1,1),
                                      no_bias=True, workspace=workspace, name=name + '_conv2')
        if dim_match:
            shortcut = data
        else:
            shortcut = mx.sym.Convolution(data=act1, num_filter=num_filter, kernel=(1,1), stride=stride, no_bias=True,
                                            workspace=workspace, name=name+'_sc')
        if memonger:
            shortcut._set_attr(mirror_stage='True')
        return conv2 + shortcut

def resnet(units, num_stages, filter_list, num_classes, image_shape, bottle_neck=True, bn_mom=0.9, workspace=256, memonger=False):
    global ignore_symbols
    num_unit = len(units)
    assert(num_unit == num_stages)
    data = mx.sym.Variable(name='data')
    data = mx.sym.BatchNorm(data=data, fix_gamma=True, eps=2e-5, momentum=bn_mom, name='bn_data')
    (nchannel, height, width) = image_shape
    if height <= 32:            # such as cifar10
        conv0 = mx.sym.Convolution(data=data, num_filter=filter_list[0], kernel=(3, 3), stride=(1,1), pad=(1, 1),
                                  no_bias=True, name="conv0", workspace=workspace)
        ignore_symbols.append(conv0)
        body = conv0
    else:                       # often expected to be 224 such as imagenet
        body = mx.sym.Convolution(data=data, num_filter=filter_list[0], kernel=(7, 7), stride=(2,2), pad=(3, 3),
                                  no_bias=True, name="conv0", workspace=workspace)
        body = mx.sym.BatchNorm(data=body, fix_gamma=False, eps=2e-5, momentum=bn_mom, name='bn0')
        body = mx.sym.relu(data=body, name='relu0')
        body = mx.sym.max_pool(data=body, kernel=(3, 3), stride=(2,2), pad=(1,1))

    for i in range(num_stages):
        body = residual_unit(body, filter_list[i+1], (1 if i==0 else 2, 1 if i==0 else 2), False,
                             name='stage%d_unit%d' % (i + 1, 1), bottle_neck=bottle_neck, workspace=workspace,
                             memonger=memonger)
        for j in range(units[i]-1):
            body = residual_unit(body, filter_list[i+1], (1,1), True, name='stage%d_unit%d' % (i + 1, j + 2),
                                 bottle_neck=bottle_neck, workspace=workspace, memonger=memonger)
    bn1 = mx.sym.BatchNorm(data=body, fix_gamma=False, eps=2e-5, momentum=bn_mom, name='bn1')
    relu1 = mx.sym.relu(data=bn1, name='relu1')
    # Although kernel is not used here when global_pool=True, we should put one
    pool1 = mx.symbol.Pooling(data=relu1, global_pool=True, kernel=(7, 7), pool_type='avg', name='pool1')
    flat = mx.symbol.Flatten(data=pool1)
    fc1 = mx.symbol.FullyConnected(data=flat, num_hidden=num_classes, no_bias=True, name='fc1')
    return mx.symbol.SoftmaxOutput(data=fc1, name='softmax')

def get_symbol(num_classes, num_layers, image_shape, conv_workspace=256, **kwargs):
    image_shape = [int(l) for l in image_shape.split(',')]
    (nchannel, height, width) = image_shape
    if height <= 28:
        num_stages = 3
        if (num_layers-2) % 9 == 0 and num_layers >= 164:
            per_unit = [(num_layers-2)//9]
            filter_list = [16, 64, 128, 256]
            bottle_neck = True
        elif (num_layers-2) % 6 == 0 and num_layers < 164:
            per_unit = [(num_layers-2)//6]
            filter_list = [16, 16, 32, 64]
            bottle_neck = False
        else:
            raise ValueError("no experiments done on num_layers {}, you can do it youself".format(num_layers))
        units = per_unit * num_stages
    else:
        if num_layers >= 50:
            filter_list = [64, 256, 512, 1024, 2048]
            bottle_neck = True
        else:
            filter_list = [64, 64, 128, 256, 512]
            bottle_neck = False
        num_stages = 4
        if num_layers == 18:
            units = [2, 2, 2, 2]
        elif num_layers == 34:
            units = [3, 4, 6, 3]
        elif num_layers == 50:
            units = [3, 4, 6, 3]
        elif num_layers == 101:
            units = [3, 4, 23, 3]
        elif num_layers == 152:
            units = [3, 8, 36, 3]
        elif num_layers == 200:
            units = [3, 24, 36, 3]
        elif num_layers == 269:
            units = [3, 30, 48, 8]
        else:
            raise ValueError("no experiments done on num_layers {}, you can do it youself".format(num_layers))

    return resnet(units       = units,
                  num_stages  = num_stages,
                  filter_list = filter_list,
                  num_classes = num_classes,
                  image_shape = image_shape,
                  bottle_neck = bottle_neck,
                  workspace   = conv_workspace)



# parse args
parser = argparse.ArgumentParser(description="train cifar10",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.set_defaults(
    # network
    network        = 'resnet',
    num_layers     = layer,
    # data
    num_classes    = 10,
    num_examples  = 50000,
    image_shape    = '3,28,28',
    pad_size       = 4,
    # train
    batch_size     = 128,
    num_epochs     = 300,
    lr             = .05,
    lr_step_epochs = '200,250',
)
args = parser.parse_args()

# load network
sym = get_symbol(**vars(args))

# download data if necessary
def _download(data_dir):
    if not os.path.isdir(data_dir):
        os.system("mkdir " + data_dir)
    cwd = os.path.abspath(os.getcwd())
    os.chdir(data_dir)
    if (not os.path.exists('train.rec')) or \
       (not os.path.exists('test.rec')) :
        import urllib, zipfile, glob
        dirname = os.getcwd()
        zippath = os.path.join(dirname, "cifar10.zip")
        urllib.urlretrieve("http://data.mxnet.io/mxnet/data/cifar10.zip", zippath)
        zf = zipfile.ZipFile(zippath, "r")
        zf.extractall()
        zf.close()
        os.remove(zippath)
        for f in glob.glob(os.path.join(dirname, "cifar", "*")):
            name = f.split(os.path.sep)[-1]
            os.rename(f, os.path.join(dirname, name))
        os.rmdir(os.path.join(dirname, "cifar"))
    os.chdir(cwd)

# data
def get_iterator(data_dir):
    data_shape = (3, 28, 28)
    if os.name == "nt":
        data_dir = data_dir[:-1] + "\\"
    if '://' not in data_dir:
        _download(data_dir)

    train = mx.io.ImageRecordIter(
        path_imgrec = data_dir + "train.rec",
        mean_img    = data_dir + "mean.bin",
        data_shape  = data_shape,
        batch_size  = batch_size,
        rand_crop   = True,
        rand_mirror = True)

    val = mx.io.ImageRecordIter(
        path_imgrec = data_dir + "test.rec",
        mean_img    = data_dir + "mean.bin",
        rand_crop   = False,
        rand_mirror = False,
        data_shape  = data_shape,
        batch_size  = batch_size)

    return (train, val)


(train_iter, val_iter) = get_iterator(data_dir)

# create a trainable module on GPU 0
model = mx.mod.Module(symbol=sym, context=ctx)
if not INFERENCE:
    print('start training')
    model.fit(train_iter,
              eval_data=val_iter,
              optimizer='sgd',
              optimizer_params={'learning_rate':0.1},
              eval_metric='acc',
              batch_end_callback = mx.callback.Speedometer(batch_size, 100),
              num_epoch=10)
    model.save_checkpoint(checkpoint_path, layer)
else:
    print('inference, load checkpoint')
    _, arg_params, aux_params = mx.model.load_checkpoint(checkpoint_path, layer)
    model.bind(data_shapes=train_iter.provide_data, label_shapes=train_iter.provide_label)
    model.set_params(arg_params=arg_params, aux_params=aux_params)


test_iter = val_iter
# predict accuracy for net
acc = mx.metric.Accuracy()
print('Accuracy: {}%'.format(model.score(test_iter, acc)[0][1]*100))

quantized_sym = quantize_graph(sym, ignore_symbols=ignore_symbols)
# print(quantized_sym.debug_str())
params = model.get_params()

def test(symbol):
    model = mx.model.FeedForward(
        symbol,
        ctx=ctx,
        arg_params=params[0],
        aux_params=params[1])
    print 'Accuracy:', model.score(test_iter)*100, '%'

print('origin:')
test(sym)
print('after quantization:')
test(quantized_sym)
