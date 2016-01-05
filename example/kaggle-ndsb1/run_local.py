import mxnet as mx
import numpy as np
import logging

# Example performance:
# INFO:root:Epoch[34] Train-accuracy=0.601388
# INFO:root:Epoch[34] Validation-accuracy=0.620949

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

# running device
dev = mx.gpu()
# batch size and input shape
batch_size = 64
data_shape = (3, 36, 36)
# training data info for learning rate reduction
num_examples = 20000
epoch_size = num_examples / batch_size
lr_factor_epoch = 15
# model saving parameter
model_prefix = "./models/sample_net"

# train data iterator
train = mx.io.ImageRecordIter(
        path_imgrec = "tr.rec",
        mean_r      = 128,
        mean_g      = 128,
        mean_b      = 128,
        scale       = 0.0078125,
        max_aspect_ratio = 0.35,
        data_shape  = data_shape,
        batch_size  = batch_size,
        rand_crop   = True,
        rand_mirror = True)

# validate data iterator
val = mx.io.ImageRecordIter(
        path_imgrec = "va.rec",
        mean_r      = 128,
        mean_b      = 128,
        mean_g      = 128,
        scale       = 0.0078125,
        rand_crop   = False,
        rand_mirror = False,
        data_shape  = data_shape,
        batch_size  = batch_size)

# network definition
# stage 1
net = mx.sym.Variable("data")
net = mx.sym.Convolution(data=net, kernel=(5, 5), num_filter=32, pad=(2, 2))
net = mx.sym.Activation(data=net, act_type="relu")
net = mx.sym.Convolution(data=net, kernel=(5, 5), num_filter=64, pad=(2, 2))
net = mx.sym.Activation(data=net, act_type="relu")
net = mx.sym.Pooling(data=net, pool_type="max", kernel=(3, 3), stride=(2, 2))
# stage 2
net = mx.sym.Convolution(data=net, kernel=(3, 3), num_filter=64, pad=(1, 1))
net = mx.sym.Activation(data=net, act_type="relu")
net = mx.sym.Convolution(data=net, kernel=(3, 3), num_filter=64, pad=(1, 1))
net = mx.sym.Activation(data=net, act_type="relu")
net = mx.sym.Convolution(data=net, kernel=(3, 3), num_filter=128, pad=(1, 1))
net = mx.sym.Activation(data=net, act_type="relu")
net = mx.sym.Pooling(data=net, pool_type="max", kernel=(3, 3), stride=(2, 2))
# stage 3
net = mx.sym.Convolution(data=net, kernel=(3, 3), num_filter=256, pad=(1, 1))
net = mx.sym.Activation(data=net, act_type="relu")
net = mx.sym.Convolution(data=net, kernel=(3, 3), num_filter=256, pad=(1, 1))
net = mx.sym.Activation(data=net, act_type="relu")
net = mx.sym.Pooling(data=net, pool_type="avg", kernel=(9, 9), stride=(1, 1))
# stage 4
net = mx.sym.Flatten(data=net)
net = mx.sym.Dropout(data=net, p=0.25)
net = mx.sym.FullyConnected(data=net, num_hidden=121)
net = mx.symbol.SoftmaxOutput(data=net, name='softmax')

# Model parameter
# This model will reduce learning rate by factor 0.1 for every 15 epoch
model = mx.model.FeedForward(
        ctx                = dev,
        symbol             = net,
        num_epoch          = 35,
        learning_rate      = 0.01,
        momentum           = 0.9,
        wd                 = 0.0001,
        clip_gradient      = 5,
        lr_scheduler       = mx.lr_scheduler.FactorScheduler(step=epoch_size * lr_factor_epoch, factor = 0.1),
        initializer        = mx.init.Xavier(factor_type="in", magnitude=2.34))

# fit the model
model.fit(
        X                  = train,
        eval_data          = val,
        batch_end_callback = mx.callback.Speedometer(batch_size, 50),
        epoch_end_callback = mx.callback.do_checkpoint(model_prefix))

