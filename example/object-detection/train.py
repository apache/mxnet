import argparse
import logging
import time
import mxnet as mx
from mxnet import gluon
from mxnet import autograd as ag
from dataset.dataloader import DataLoader
from dataset import VOCDetection
from dataset import transform
from config import config as cfg
from trainer.trainer import train_ssd
from model_zoo.ssd import ssd_512_resnet18_v1
from block.loss import *
from block.target import *

logging.basicConfig(level=logging.DEBUG)


def ctx_as_list(ctx):
    if isinstance(ctx, mx.Context):
        ctx = [ctx]
    return ctx

# dataset
transform = transform.SSDAugmentation((300, 300))
train_dataset = VOCDetection('./data/VOCdevkit', [(2007, 'trainval')], transform=transform)
val_dataset = VOCDetection('./data/VOCdevkit', [(2007, 'test')], transform=transform)
train_data = DataLoader(train_dataset, 32, True, last_batch='rollover')
val_data = DataLoader(val_dataset, 2, False, last_batch='keep')
target_generator = SSDTargetGenerator()
# for data in train_data:
#     pass
#     print(data[0].shape, data[1].shape, type(data[0]), type(data[1]))
    # import cv2
    # import numpy as np
    # img = data[0][0].asnumpy()[:, :, (2, 1, 0)].astype('uint8')
    # cv2.imshow('debug', img)
    # cv2.waitKey()

# network
net = ssd_512_resnet18_v1(pretrained=(0, 0))

lr = 0.0001
wd = 0.00005
momentum = 0.9
log_interval = 1

# training process
def train(net, train_data, val_data, epochs, ctx=mx.cpu()):
    ctx = ctx_as_list(ctx)
    net.initialize(mx.init.Uniform(), ctx=ctx)
    trainer = gluon.Trainer(net.collect_params(), 'sgd',
        {'learning_rate': lr, 'wd': wd, 'momentum':momentum})
    metric = None
    loss = None

    for epoch in range(epochs):
        tic = time.time()
        btic = time.time()
        for i, batch in enumerate(train_data):
            data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0)
            label = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0)
            outputs = []
            Ls = []
            with ag.record():
                for x, y in zip(data, label):
                    z = net(x)
                    target_generator(z, y)
                    L = None
                    Ls.append(L)
                    outputs.append(z)
                for L in Ls:
                    pass
                    # L.backward()
            # trainer.step(batch.data[0].shape[0])
            # metric.update(label, outputs)
            if log_interval and not (i + 1) % log_interval:
                # name, acc = metric.get()
                logging.info("Epoch [%d] Batch [%d]"%(epoch, i))

train(net, train_data, val_data, 10)
