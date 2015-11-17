import os
import sys
curr_path = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(curr_path, "../network"))
import lenet
import data
import mxnet as mx
import logging
logging.basicConfig(level=logging.DEBUG)

train, val = data.get_iterator(
    batch_size  = 100,
    input_shape = (1, 28, 28))

model = mx.model.FeedForward.create(
    ctx                = mx.gpu(),
    symbol             = lenet.get_symbol(10),
    num_epoch          = 20,
    learning_rate      = 0.05,
    momentum           = 0.9,
    wd                 = 0.00001,
    X                  = train,
    eval_data          = val,
    batch_end_callback = mx.callback.Speedometer(100))
