"""Target generator for training.
Target generator is used to generate training targets, given anchors, ground-truths,
match results and sampler.
"""
from mxnet import nd
from mxnet import autograd
from mxnet.gluon import Block
from block.matcher import CompositeMatcher, BipartiteMatcher, MaximumMatcher
from block.sampler import NaiveSampler, OHEMSampler
from block.coder import MultiClassEncoder, NormalizedBoxCenterEncoder
import numpy as np

class SSDTargetGenerator(Block):
    """

    """
    def __init__(self, threshold=0.5, **kwargs):
        super(SSDTargetGenerator, self).__init__(**kwargs)
        self._matcher = CompositeMatcher([BipartiteMatcher(), MaximumMatcher(threshold)])
        # self._sampler = NaiveSampler()
        self._sampler = OHEMSampler(3, thresh=0.5)
        self._cls_encoder = MultiClassEncoder()
        self._box_encoder = NormalizedBoxCenterEncoder()

    def forward(self, predictions, labels):
        # predictions: [cls_preds, box_preds, anchors]
        anchors = predictions[2].reshape((-1, 4))
        gt_boxes = nd.slice_axis(labels, axis=-1, begin=1, end=5)
        gt_ids = nd.slice_axis(labels, axis=-1, begin=0, end=1)
        # print(anchors)
        # print(gt_boxes)
        # print(anchors.shape, gt_boxes.shape)
        ious = nd.transpose(nd.contrib.box_iou(anchors, gt_boxes), (1, 0, 2))
        # print(ious.shape)
        matches = self._matcher(ious)
        # d = matches[0].asnumpy()
        # import numpy as np
        # print(np.where(d>=0)[0])
        # d = d[np.where(d >= 0)[0]]
        # print(d)
        # ious2 = ious[0].asnumpy()
        # print(np.sum(ious2 > 0))
        # print(np.amax(ious2))
        samples = self._sampler(matches, predictions[0], ious)
        # from trainer.debugger import super_print
        # super_print(samples[0].asnumpy())
        # raise
        cls_targets = self._cls_encoder(samples, matches, gt_ids)
        # print('cls-targets', cls_targets[0])
        box_targets, box_masks = self._box_encoder(samples, matches, anchors, gt_boxes)
        # print('box-targets', box_targets[0], 'box-masks', box_masks[0])
        # ref = nd.contrib.MultiBoxTarget(*[predictions[2], labels, predictions[0].transpose(axes=(0, 2, 1))], negative_mining_ratio=3)
        # loc_target, loc_mask, ref_cls_target = ref
        # print('diff', np.sum(np.abs(ref_cls_target.asnumpy().flatten() - cls_targets.asnumpy().flatten())))
        # print('diff2', np.sum(np.abs(loc_target.asnumpy().flatten() - box_targets.asnumpy().flatten())))
        # print('diff3', np.sum(np.abs(loc_mask.asnumpy().flatten() - box_masks.asnumpy().flatten())))
        # raise
        # super_print(np.array([ref_cls_target[0].asnumpy(), cls_targets[0].asnumpy()]).transpose((1, 0)))
        # print('where', np.where(ref_cls_target[0].asnumpy() == 0))
        # print('where2', np.where(cls_targets[0].asnumpy() == 0))
        # raise
        return cls_targets, box_targets, box_masks
