import argparse
import logging
import time
import mxnet as mx
from mxnet import nd
from mxnet import gluon
from mxnet import autograd as ag
from dataset.dataloader import DataLoader
from dataset import VOCDetection
from dataset import transform
from config import config as cfg
from trainer.trainer import train_ssd
from model_zoo.ssd import *
from block.loss import *
from block.target import *
from block.loss import *
from trainer.metric import Accuracy, SmoothL1
from trainer.debuger import super_print, find_abnormal

logging.basicConfig(level=logging.DEBUG)


def ctx_as_list(ctx):
    if isinstance(ctx, mx.Context):
        ctx = [ctx]
    return ctx

# dataset
num_class = 20
transform = transform.SSDAugmentation((512, 512))
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
net = ssd_512_resnet18_v1(classes=num_class, pretrained=(0, 0))

lr = 0.01
wd = 0.00005
momentum = 0.9
log_interval = 1
dtype = 'float32'

# monitor
# print(net.collect_params())
# raise
# checker = net.collect_params()['conv0_weight']
checker = net.collect_params()['stage3_conv1_weight']

# training process
def train(net, train_data, val_data, epochs, ctx=mx.cpu()):
    ctx = ctx_as_list(ctx)
    net.initialize(mx.init.Uniform(), ctx=ctx)
    trainer = gluon.Trainer(net.collect_params(), 'sgd',
        {'learning_rate': lr, 'wd': wd, 'momentum':momentum})
    cls_loss = FocalLoss(num_class=num_class+1)
    box_loss = gluon.loss.L1Loss()
    cls_metric = Accuracy(axis=-1)
    box_metric = SmoothL1()

    for epoch in range(epochs):
        tic = time.time()
        btic = time.time()
        for i, batch in enumerate(train_data):
            data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0)
            label = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0)
            outputs = []
            labels = []
            box_preds = []
            box_labels = []
            Ls = []
            with ag.record():
                for x, y in zip(data, label):
                    x = nd.cast(x, dtype)
                    y = nd.cast(y, dtype)
                    z = net(x)
                    cls_targets, box_targets, box_masks = target_generator(z, y)
                    # super_print(y, cls_targets, box_targets)
                    # raise
                    loss1 = cls_loss(z[0], cls_targets)
                    loss2 = box_loss((z[1] - box_targets) * box_masks, nd.zeros_like(box_targets))
                    L = loss1 + loss2
                    # L = loss1
                    Ls.append(L)
                    outputs.append(z[0])
                    labels.append(cls_targets)
                    box_preds.append(z[1] * box_masks)
                    box_labels.append(box_targets)
                for L in Ls:
                    pass
                    L.backward()
            trainer.step(batch[0].shape[0], ignore_stale_grad=True)
            cls_metric.update(labels, outputs)
            box_metric.update(box_labels, box_preds)
            if log_interval and not (i + 1) % log_interval:
                # print(checker.grad())
                name, acc = cls_metric.get()
                name1, mae = box_metric.get()
                logging.info("Epoch [%d] Batch [%d], %s=%f, %s=%f"%(epoch, i, name, acc, name1, mae))

ctx = [mx.gpu(i) for i in range(8)]
train(net, train_data, val_data, 10, ctx=ctx)
