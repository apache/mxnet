"""Target generator for training.
Target generator is used to generate training targets, given anchors, ground-truths,
match results and sampler.
"""
from mxnet import nd
from mxnet.gluon import Block
from block.matcher import CompositeMatcher, BipartiteMatcher, MaximumMatcher

class SSDTargetGenerator(Block):
    """

    """
    def __init__(self, threshold=0.5, **kwargs):
        super(SSDTargetGenerator, self).__init__(**kwargs)
        self._matcher = CompositeMatcher([BipartiteMatcher(), MaximumMatcher(threshold)])

    def forward(self, predictions, labels):
        # predictions: [cls_preds, box_preds, anchors]
        ious = nd.contrib.box_iou(predictions[2], nd.slice_axis(axis=2, begin=1, end=5))
        print(ious)
        matches = self._matcher(ious)
        print(matches)
        raise
