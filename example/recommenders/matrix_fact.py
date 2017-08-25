import math
import mxnet as mx
import numpy as np
import mxnet.notebook.callback

import logging
logging.basicConfig(level=logging.DEBUG)

def RMSE(label, pred):
    ret = 0.0
    n = 0.0
    pred = pred.flatten()
    for i in range(len(label)):
        ret += (label[i] - pred[i]) * (label[i] - pred[i])
        n += 1.0
    return math.sqrt(ret / n)


def train(network, data_pair, num_epoch, learning_rate, optimizer='sgd', opt_args=None, ctx=[mx.gpu(0)]):
    np.random.seed(123)  # Fix random seed for consistent demos
    mx.random.seed(123)  # Fix random seed for consistent demos
    if not opt_args:
        opt_args = {}
    if optimizer=='sgd' and (not opt_args):
        opt_args['momentum'] = 0.9

    model = mx.model.FeedForward(
        ctx = ctx,
        symbol = network,
        num_epoch = num_epoch,
        optimizer = optimizer,
        learning_rate = learning_rate,
        wd = 1e-4,
        **opt_args
    )

    train, test = (data_pair)

    lc = mxnet.notebook.callback.LiveLearningCurve('RMSE', 1)
    model.fit(X = train,
              eval_data = test,
              eval_metric = RMSE,
              **mxnet.notebook.callback.args_wrapper(lc)
              )
    return lc
