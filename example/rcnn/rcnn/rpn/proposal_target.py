"""
Proposal Operator transform anchor coordinates into ROI coordinates with prediction results on
classification probability and bounding box prediction results, and image size and scale information.
"""

import mxnet as mx
import numpy as np
import numpy.random as npr
from rcnn.config import config
from helper.processing.bbox_regression import bbox_overlaps
from helper.processing.bbox_regression import expand_bbox_regression_targets
from helper.processing.bbox_transform import bbox_transform
from helper.processing.generate_anchor import generate_anchors
import logging


DEBUG = False

class ProposalTargetOperator(mx.operator.CustomOp):
    def __init__(self, num_classes, is_train=False):
        super(ProposalTargetOperator, self).__init__()
        self._num_classes = int(num_classes)

        if DEBUG:
            self._count = 0
            self._fg_num = 0
            self._bg_num = 0
        if is_train:
            self.cfg_key = 'TRAIN'
        else:
            self.cfg_key = 'TEST'
        self._img_per_batch = 1

    def forward(self, is_train, req, in_data, out_data, aux):
        assert config.TRAIN.BATCH_SIZE % self._img_per_batch == 0, \
                'IMAGESPERBATCH {} must devide BATCHSIZE {}'.format(self._img_per_batch, config.TRAIN.BATCH_SIZE)
        num_images = self._img_per_batch  # 1
        assert num_images == 1, "only support signle image"
        rois_per_image = config.TRAIN.BATCH_SIZE / self._img_per_batch
        fg_rois_per_image = np.round(config.TRAIN.FG_FRACTION * rois_per_image).astype(int)  # neg : pos=3 : 1
        all_rois = in_data[0].asnumpy()
        gt_boxes = in_data[1].asnumpy()
        gt_boxes = gt_boxes[np.where(gt_boxes[:, :5].mean(axis=1) != -1)]

        # Include ground-truth boxes in the set of candidate rois
        zeros = np.zeros((gt_boxes.shape[0], 1), dtype=gt_boxes.dtype)
        all_rois = np.vstack(
            (all_rois, np.hstack((zeros, gt_boxes[:, :-1])))
        )
        # Sanity check: single batch only
        assert np.all(all_rois[:, 0] == 0), \
                'Only single item batches are supported'

        # Sample rois with classification labels and bounding box regression
        # targets
        labels, rois, bbox_targets, bbox_inside_weights = _sample_rois(
            all_rois, gt_boxes, fg_rois_per_image,
            rois_per_image, self._num_classes, self.cfg_key)

        if DEBUG:
            print "labels=", labels
            print 'num fg: {}'.format((labels > 0).sum())
            print 'num bg: {}'.format((labels == 0).sum())
            self._count += 1
            self._fg_num += (labels > 0).sum()
            self._bg_num += (labels == 0).sum()
            print "self._count=", self._count
            print 'num fg avg: {}'.format(self._fg_num / self._count)
            print 'num bg avg: {}'.format(self._bg_num / self._count)
            print 'ratio: {:.3f}'.format(float(self._fg_num) / float(self._bg_num))

        self.assign(out_data[0], req[0], rois)
        self.assign(out_data[1], req[1], labels)
        self.assign(out_data[2], req[2], bbox_targets)
        self.assign(out_data[3], req[3], bbox_inside_weights)
        self.assign(out_data[4], req[4], np.array(bbox_inside_weights > 0).astype(np.float32) ) # no normalization

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        self.assign(in_grad[0], req[0], 0)
        self.assign(in_grad[1], req[1], 0)

@mx.operator.register("proposal_target")
class ProposalTargetProp(mx.operator.CustomOpProp):
    def __init__(self, num_classes, is_train=False):
        super(ProposalTargetProp, self).__init__(need_top_grad=False)
        self._num_classes = int(num_classes)
        self._is_train = True if is_train == 'True' else False
        if self._is_train:
            self.cfg_key = 'TRAIN'
        else:
            self.cfg_key = 'TEST'
        self._img_per_batch = 1

    def list_arguments(self):
        return ['rpn_roi', 'gt_boxes']

    def list_outputs(self):
        return ['roi', 'label', 'bbox_target', 'bbox_inside_weight', 'bbox_outside_weight']

    def infer_shape(self, in_shape):
        rpn_roi_shape = in_shape[0]
        gt_boxes_shape = in_shape[1]

        batch_size = config.TRAIN.BATCH_SIZE / self._img_per_batch
        # output shape
        roi_shape = (batch_size, 5)  # used for input of roi-pooling
        label_shape = (batch_size, )  # becauseful not set (batch_size, 1)
        bbox_target_shape = (batch_size, self._num_classes * 4)
        bbox_inside_weight_shape = (batch_size, self._num_classes * 4)
        bbox_outside_weight_shape = (batch_size, self._num_classes * 4)

        return [rpn_roi_shape, gt_boxes_shape], [roi_shape, label_shape, bbox_target_shape, bbox_inside_weight_shape, bbox_outside_weight_shape]

    def create_operator(self, ctx, shapes, dtypes):
        return ProposalTargetOperator(self._num_classes, self._is_train)

    def declare_backward_dependency(self, out_grad, in_data, out_data):
        return []

def _compute_targets(ex_rois, gt_rois, labels):
    """Compute bounding-box regression targets for an image."""

    assert ex_rois.shape[0] == gt_rois.shape[0]
    assert ex_rois.shape[1] == 4
    assert gt_rois.shape[1] == 4

    targets = bbox_transform(ex_rois, gt_rois)
    if config.TRAIN.BBOX_NORMALIZATION_PRECOMPUTED:
        # Optionally normalize targets by a precomputed mean and stdev
        targets = ((targets - np.array(config.TRAIN.BBOX_MEANS))
                / np.array(config.TRAIN.BBOX_STDS))
    return np.hstack(
            (labels[:, np.newaxis], targets)).astype(np.float32, copy=False)

def _sample_rois(all_rois, gt_boxes, fg_rois_per_image, rois_per_image, num_classes, key):
    """Generate a random sample of RoIs comprising foreground and background
    examples.
    """
    # overlaps: (rois x gt_boxes)
    overlaps = bbox_overlaps(
        np.ascontiguousarray(all_rois[:, 1:5], dtype=np.float),
        np.ascontiguousarray(gt_boxes[:, :4], dtype=np.float))
    gt_assignment = overlaps.argmax(axis=1)
    max_overlaps = overlaps.max(axis=1)
    labels = gt_boxes[gt_assignment, 4]

    # Select foreground RoIs as those with >= FG_THRESH overlap
    fg_inds = np.where(max_overlaps >= config.TRAIN.FG_THRESH)[0]
    # Guard against the case when an image has fewer than fg_rois_per_image
    # foreground RoIs
    fg_rois_per_this_image = min(fg_rois_per_image, fg_inds.size)
    # Sample foreground regions without replacement
    if fg_inds.size > 0:
        fg_inds = npr.choice(fg_inds, size=fg_rois_per_this_image, replace=False)
    if fg_inds.size < fg_rois_per_image:
        fg_inds_ = npr.choice(fg_inds, size=fg_rois_per_image-fg_inds.size, replace=True)
        fg_inds = np.hstack((fg_inds_, fg_inds))

    # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
    bg_inds_ = np.where((max_overlaps < config.TRAIN.BG_THRESH_HI) &
                       (max_overlaps >= config.TRAIN.BG_THRESH_LO))[0]
    if len(bg_inds_) == 0 and key == 'TRAIN':
        bg_inds = np.where((max_overlaps < config.TRAIN.BG_THRESH_HI+0.2) &
                       (max_overlaps >= 0))[0]
    else:
        bg_inds = bg_inds_

    if len(bg_inds) == 0:
        logging.log(logging.ERROR, "currently len(bg_inds) is zero")

    # Compute number of background RoIs to take from this image (guarding
    # against there being fewer than desired)
    bg_rois_per_this_image = rois_per_image - len(fg_inds)
    bg_rois_per_this_image = min(bg_rois_per_this_image, bg_inds.size)
    # Sample background regions without replacement
    if bg_inds.size > 0:
        bg_inds = npr.choice(bg_inds, size=bg_rois_per_this_image, replace=False)
    if bg_inds.size < rois_per_image-fg_rois_per_image:
        bg_inds_ = npr.choice(bg_inds, size=rois_per_image-fg_rois_per_image-bg_inds.size, replace=True)
        bg_inds = np.hstack((bg_inds_, bg_inds))

    # The indices that we're selecting (both fg and bg)
    keep_inds = np.append(fg_inds, bg_inds)
    # Select sampled values from various arrays:
    labels = labels[keep_inds]
    # Clamp labels for the background RoIs to 0
    labels[fg_rois_per_this_image:] = 0
    rois = all_rois[keep_inds]

    bbox_target_data = _compute_targets(
        rois[:, 1:5], gt_boxes[gt_assignment[keep_inds], :4], labels)

    bbox_targets, bbox_inside_weights = \
        expand_bbox_regression_targets(bbox_target_data, num_classes)

    return labels, rois, bbox_targets, bbox_inside_weights