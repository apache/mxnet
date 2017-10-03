"""Detailed training processes for different algorithms."""
import argparse
from mxnet import gluon
from mxnet import ndarray as nd
from mxnet.gluon import Block

def train_ssd(net, x, y, sampler, matcher, class_encoder, box_encoder,
              class_loss, box_loss, class_weight=1.0, box_weight=1.0):
    """
    """
    cls_preds, box_preds, anchors = net(x)
    gt_boxes = nd.slice_axis(y, begin=1, end=5, axis=2)
    gt_classes = nd.slice_axis(y, begin=0, end=5, axis=2)
    ious = nd.contrib.box_iou(anchors.reshape(0, -1, 4), gt_boxes)
    matches = matcher(ious)
    sampler = sampler(matches)
    box_codecs, masks = box_encoder(sampler, matches, anchors, gt_boxes)
    class_codecs = class_encoder(sampler, matches, gt_classes)
    closs = class_loss(cls_preds, class_codecs)
    bloss = box_loss(box_preds * masks, box_codecs)
    loss = class_weight * closs + box_weight * bloss
    return loss
