import argparse
import os
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
import model_zoo
from block.loss import *
from block.target import *
from block.coder import MultiClassDecoder, NormalizedBoxCenterDecoder
from trainer.metric import Accuracy, SmoothL1, LossRecorder, MultiBoxMetric
from trainer.debugger import super_print, find_abnormal
from evaluation.eval_metric import VOC07MApMetric, MApMetric

def train_net(model, dataset, data_shape, batch_size, end_epoch, lr, momentum, wd, log_interval=50,
              lr_steps=[], lr_factor=1.,
              pretrained=0, seed=None, log_file=None, dev=False, ctx=mx.cpu(), **kwargs):
    """Wrapper function for entire training phase.




    """
    if dev:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    if log_file:
        logger = logging.getLogger()
        fh = logging.FileHandler(log_file)
        logger.addHandler(fh)

    if isinstance(seed, int) and seed > 0:
        random.seed(seed)

    data_shape = [int(x) for x in data_shape.split(',')]
    if len(data_shape) == 1:
        data_shape = data_shape * 2

    if dataset == 'voc':
        # dataset
        num_class = 20
        t = transform.SSDAugmentation(data_shape)
        t2 = transform.SSDValid(data_shape)
        train_dataset = VOCDetection('./data/VOCdevkit', [(2007, 'trainval'), (2012, 'trainval')], transform=t)
        # train_dataset = VOCDetection('./data/VOCdevkit', [(2007, 'train')], transform=transform)
        val_dataset = VOCDetection('./data/VOCdevkit', [(2007, 'test')], transform=t2)
    else:
        raise NotImplementedError("Dataset {} not supported.".format(dataset))

    train_data = DataLoader(train_dataset, batch_size, True, last_batch='rollover')
    val_data = DataLoader(val_dataset, batch_size, False, last_batch='keep')

    net = model_zoo.get_detection_model(model, pretrained=pretrained, classes=num_class)
    if dev:
        print(net)

    def ctx_as_list(ctx):
        if isinstance(ctx, mx.Context):
            ctx = [ctx]
        return ctx

    if not isinstance(lr_factor, list):
        lr_factor = [lr_factor]
    if len(lr_factor) == 1 and len(lr_steps) > 1:
        lr_factor *= len(lr_steps)

    # logging.debug(str(val_dataset))
    # for data in train_data:
    #     import cv2
    #     import numpy as np
    #     for i in range(data[0].shape[0]):
    #         img = data[0][i].asnumpy().transpose((1, 2, 0))[:, :, (2, 1, 0)].astype('uint8')
    #         w, h, _ =  img.shape
    #         label = data[1][i].asnumpy()
    #         canvas = np.asarray(img.copy())
    #         for j in range(label.shape[0]):
    #             if label[j, 0] < 0:
    #                 break
    #             pt1 = (int(label[j, 1] * w), int(label[j, 2] * h))
    #             pt2 = (int(label[j, 3] * w), int(label[j, 4] * w))
    #             cv2.rectangle(canvas, pt1, pt2, (255, 0, 0), 2)
    #         cv2.imshow('debug', canvas)
    #         cv2.waitKey()




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
        valid_metric = VOC07MApMetric(class_names=val_dataset.classes)
        for i, batch in enumerate(val_data):
            data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0)
            label = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0)
            for x, y in zip(data, label):
                z = net(x)
                cls_preds, box_preds, anchors = z
                # print('box_preds',box_preds)
                # print('anchors', anchors)
                # boxes = box_decoder(box_preds, anchors)
                # boxes = nd.clip(boxes, 0.0, 1.0)
                # cls_ids, scores = cls_decoder(nd.softmax(cls_preds))
                # result = nd.concat(cls_ids.reshape((0, 0, 1)), scores.reshape((0, 0, 1)), boxes, dim=2)
                # print(boxes)
                # print(cls_ids)
                # print(scores)
                # print(result)
                # out = nd.contrib.box_nms(result, topk=400)
                out = mx.nd.contrib.MultiBoxDetection(nd.softmax(cls_preds).transpose((0, 2, 1)), box_preds.reshape((0, -1)), anchors, nms_topk=400)
                # print(out)
                # results.append(out.asnumpy())
                valid_metric.update([y], [out])
        # results = np.vstack(results)
        # write to disk for eval
        return valid_metric.get()
        # return val_dataset.eval_results(results)


    # training process
    def train(net, train_data, val_data, epochs, ctx=mx.cpu()):
        ctx = ctx_as_list(ctx)
        target_generator = SSDTargetGenerator()
        box_weight = None
        net.initialize(mx.init.Uniform(), ctx=ctx)
        net.collect_params().reset_ctx(ctx)
        net.hybridize()
        trainer = gluon.Trainer(net.collect_params(), 'sgd',
            {'learning_rate': lr, 'wd': wd, 'momentum':momentum})
        # cls_loss = FocalLoss(num_class=(num_class+1), weight=1.0)
        cls_loss = SoftmaxCrossEntropyLoss(size_average=False)
        # box_loss = gluon.loss.L1Loss()
        box_loss = SmoothL1Loss(weight=box_weight, size_average=False)
        cls_metric = Accuracy(axis=-1, ignore_label=-1)
        box_metric = SmoothL1()
        cls_metric1 = LossRecorder('CrossEntropy')
        box_metric1 = LossRecorder('SmoothL1Loss')
        # debug_metric = MultiBoxMetric()

        for epoch in range(epochs):
            if epoch in lr_steps:
                new_lr = trainer.learning_rate * lr_factor[lr_steps.index(epoch)]
                trainer.set_learning_rate(new_lr)
                logging.info("[Epoch {}] Set learning rate to {}".format(epoch, new_lr))
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
                        # x = nd.cast(x, dtype)
                        # y = nd.cast(y, dtype)
                        z = net(x)
                        with ag.pause():
                            cls_targets, box_targets, box_masks = target_generator(z, y)
                            valid_cls = nd.sum(cls_targets >= 0, axis=0, exclude=True)
                            valid_cls = nd.maximum(valid_cls, nd.ones_like(valid_cls))
                            valid_box = nd.sum(box_masks > 0, axis=0, exclude=True)
                        # super_print(y, cls_targets, box_targets)
                        # raise
                        loss1 = cls_loss(z[0], cls_targets)
                        # valid_cls1 = nd.sum(valid_cls).asscalar()
                        # print(valid_cls1)
                        # loss1 = loss1 * cls_targets.shape[1] / valid_cls
                        loss1 = loss1 / valid_cls
                        loss2 = box_loss(z[1] * box_masks, box_targets)
                        # loss2 = loss2 * box_masks.shape[1] / valid_cls
                        loss2 = loss2 / valid_box
                        L = loss1 + loss2
                        # L = loss1
                        Ls.append(L)
                        outputs.append(z[0])
                        labels.append(cls_targets)
                        box_preds.append(z[1] * box_masks)
                        box_labels.append(box_targets)
                        losses1.append(loss1)
                        losses2.append(loss2)
                    ag.backward(Ls)
                batch_size = batch[0].shape[0]
                trainer.step(batch_size, ignore_stale_grad=True)
                cls_metric.update(labels, outputs)
                # box_metric.update(box_labels, box_preds)
                cls_metric1.update(losses1)
                box_metric1.update(losses2)
                # debug_metric.update(cls_targets, [nd.softmax(z[0]).transpose((0, 2, 1)), nd.smooth_l1((z[1] - box_targets) * box_masks, scalar=1.0), cls_targets])
                if log_interval and not (i + 1) % log_interval:
                    # print(checker.grad())
                    name, acc = cls_metric.get()
                    name1, mae = box_metric.get()
                    name2, focalloss = cls_metric1.get()
                    name3, smoothl1loss = box_metric1.get()
                    logging.info("Epoch [%d] Batch [%d], Speed: %f samples/sec, %s=%f, %s=%f, %s=%f"%(epoch, i, batch_size/(time.time()-btic), name, acc, name2, focalloss, name3, smoothl1loss))
                    # names, values = debug_metric.get()
                    # logging.info("%s=%f, %s=%f"%(names[0], values[0], names[1], values[1]))
                btic = time.time()

            name, acc = cls_metric.get()
            name1, mae = box_metric1.get()
            name2, focalloss = cls_metric1.get()
            net.collect_params().save(os.path.join(os.path.dirname(__file__), '..', 'model', 'ssd.params'))
            logging.info('[Epoch %d] training: %s=%f, %s=%f, %s=%f'%(epoch, name, acc, name1, mae, name2, focalloss))
            logging.info('[Epoch %d] time cost: %f'%(epoch, time.time()-tic))
            map_name, mean_ap = evaluate_voc(net, val_data, ctx)
            # name, val_acc = test(ctx)
            for name, ap in zip(map_name, mean_ap):
                logging.info('[Epoch %d] validation: %s=%f'%(epoch, name, ap))

    train(net, train_data, val_data, end_epoch, ctx=ctx)
