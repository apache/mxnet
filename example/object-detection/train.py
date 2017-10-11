import argparse
import logging
import time
import random
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
from block.coder import MultiClassDecoder, NormalizedBoxCenterDecoder
from trainer.metric import Accuracy, SmoothL1, LossRecorder
from trainer.debugger import super_print, find_abnormal

# experimental stuff
logging.basicConfig(level=logging.DEBUG)
random.seed(123)
logger = logging.getLogger()
fh = logging.FileHandler('train.log')
logger.addHandler(fh)


def ctx_as_list(ctx):
    if isinstance(ctx, mx.Context):
        ctx = [ctx]
    return ctx

# dataset
num_class = 20
transform = transform.SSDAugmentation((512, 512))
train_dataset = VOCDetection('./data/VOCdevkit', [(2007, 'trainval'), (2012, 'trainval')], transform=transform)
# train_dataset = VOCDetection('./data/VOCdevkit', [(2007, 'train')], transform=transform)
val_dataset = VOCDetection('./data/VOCdevkit', [(2007, 'test')], transform=transform)
train_data = DataLoader(train_dataset, 32, True, last_batch='rollover')
val_data = DataLoader(val_dataset, 32, False, last_batch='keep')
target_generator = SSDTargetGenerator()
# logging.debug(str(val_dataset))
# for data in train_data:
#     pass
# raise
#     print(data[0].shape, data[1].shape, type(data[0]), type(data[1]))
    # import cv2
    # import numpy as np
    # img = data[0][0].asnumpy()[:, :, (2, 1, 0)].astype('uint8')
    # cv2.imshow('debug', img)
    # cv2.waitKey()

# network
net = ssd_512_resnet18_v1(classes=num_class, pretrained=(1, 0))

lr = 0.01
wd = 0.00005
momentum = 0.9
log_interval = 50
dtype = 'float32'
box_weight = 5.0

# monitor
# print(net.collect_params())
# raise
# checker = net.collect_params()['conv0_weight']
checker = net.collect_params()['stage3_conv1_weight']

box_decoder = NormalizedBoxCenterDecoder()
cls_decoder = MultiClassDecoder()

def evaluate_voc(net, val_data, ctx):
    ctx = ctx_as_list(ctx)
    results = []
    for i, batch in enumerate(val_data):
        data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0)
        for x in data:
            z = net(x)
            cls_preds, box_preds, anchors = z
            # print('box_preds',box_preds)
            # print('anchors', anchors)
            boxes = box_decoder(box_preds, anchors)
            boxes = nd.clip(boxes, 0.0, 1.0)
            cls_ids, scores = cls_decoder(nd.sigmoid(cls_preds))
            result = nd.concat(cls_ids.reshape((0, 0, 1)), scores.reshape((0, 0, 1)), boxes, dim=2)
            # print(boxes)
            # print(cls_ids)
            # print(scores)
            # print(result)
            out = nd.contrib.box_nms(result, topk=400)
            # print(out)
            results.append(out.asnumpy())
    results = np.vstack(results)
    # write to disk for eval
    return val_dataset.eval_results(results)


# training process
def train(net, train_data, val_data, epochs, ctx=mx.cpu()):
    ctx = ctx_as_list(ctx)
    net.initialize(mx.init.Uniform(), ctx=ctx)
    net.collect_params().reset_ctx(ctx)
    net.hybridize()
    trainer = gluon.Trainer(net.collect_params(), 'sgd',
        {'learning_rate': lr, 'wd': wd, 'momentum':momentum})
    cls_loss = FocalLoss(num_class=(num_class+1), weight=1.0)
    # box_loss = gluon.loss.L1Loss()
    box_loss = SmoothL1Loss(weight=4)
    cls_metric = Accuracy(axis=-1, ignore_label=0)
    box_metric = SmoothL1()
    cls_metric1 = LossRecorder('FocalLoss')
    box_metric1 = LossRecorder('SmoothL1Loss')

    for epoch in range(epochs):
        tic = time.time()
        btic = time.time()
        cls_metric.reset()
        cls_metric1.reset()
        box_metric.reset()
        box_metric1.reset()
        for i, batch in enumerate(train_data):
            data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0)
            label = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0)
            outputs = []
            labels = []
            box_preds = []
            box_labels = []
            losses1 = []
            losses2 = []
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
                    losses1.append(loss1)
                    losses2.append(loss2)
                for L in Ls:
                    pass
                    L.backward()
            batch_size = batch[0].shape[0]
            trainer.step(batch_size, ignore_stale_grad=True)
            cls_metric.update(labels, outputs)
            # box_metric.update(box_labels, box_preds)
            cls_metric1.update(losses1)
            box_metric1.update(losses2)
            if log_interval and not (i + 1) % log_interval:
                # print(checker.grad())
                name, acc = cls_metric.get()
                name1, mae = box_metric.get()
                name2, focalloss = cls_metric1.get()
                name3, smoothl1loss = box_metric1.get()
                logging.info("Epoch [%d] Batch [%d], Speed: %f samples/sec, %s=%f, %s=%f, %s=%f"%(epoch, i, batch_size/(time.time()-btic), name, acc, name2, focalloss, name3, smoothl1loss))
            btic = time.time()

        name, acc = cls_metric.get()
        name1, mae = box_metric1.get()
        name2, focalloss = cls_metric1.get()
        logging.info('[Epoch %d] training: %s=%f, %s=%f, %s=%f'%(epoch, name, acc, name1, mae, name2, focalloss))
        logging.info('[Epoch %d] time cost: %f'%(epoch, time.time()-tic))
        map_name, mean_ap = evaluate_voc(net, val_data, ctx)
        # name, val_acc = test(ctx)
        logging.info('[Epoch %d] validation: %s=%f'%(epoch, map_name, mean_ap))

ctx = [mx.gpu(i) for i in range(8)]
# ctx = mx.cpu()
train(net, train_data, val_data, 100, ctx=ctx)
