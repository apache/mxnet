# pylint: skip-file
import mxnet as mx
import numpy as np
import logging
import data
from autoencoder import AutoEncoderModel

if __name__ == '__main__':
    # set to INFO to see less information during training
    logging.basicConfig(level=logging.DEBUG)
    ae_model = AutoEncoderModel(mx.gpu(0), [784,500,500,2000,10], pt_dropout=0.2,
        internal_act='relu', output_act='relu')

    X, _ = data.get_mnist()
    train_X = X[:60000]
    val_X = X[60000:]

    ae_model.layerwise_pretrain(train_X, 256, 50000, 'sgd', l_rate=0.1, decay=0.0,
                             lr_scheduler=mx.misc.FactorScheduler(20000,0.1))
    ae_model.finetune(train_X, 256, 100000, 'sgd', l_rate=0.1, decay=0.0,
                   lr_scheduler=mx.misc.FactorScheduler(20000,0.1))
    ae_model.save('mnist_pt.arg')
    ae_model.load('mnist_pt.arg')
    print "Training error:", ae_model.eval(train_X)
    print "Validation error:", ae_model.eval(val_X)
