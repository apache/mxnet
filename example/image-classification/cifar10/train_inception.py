import os
import sys
curr_path = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(curr_path, "../network"))
import inception_bn_28
import data
import mxnet as mx
import logging
logging.basicConfig(level=logging.DEBUG)

learning_rate = .05
batch_size = 128
num_epoch = 10
num_gpus = 1

train, val = data.get_iterator(batch_size)

model = mx.model.FeedForward.create(
    ctx                = [mx.gpu(i) for i in range(num_gpus)],
    symbol             = inception_bn_28.get_symbol(10),
    num_epoch          = num_epoch,
    learning_rate      = learning_rate,
    momentum           = 0.9,
    wd                 = 0.00001,
    X                  = train,
    eval_data          = val,
    batch_end_callback = mx.callback.Speedometer(100))
