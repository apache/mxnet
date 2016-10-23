"""
Proposal Operator transform anchor coordinates into ROI coordinates with prediction results on
classification probability and bounding box prediction results, and image size and scale information.
"""

import mxnet as mx
import numpy as np
import numpy.random as npr

from rcnn.config import config
from helper.processing.generate_anchor import generate_anchors
from helper.processing.bbox_transform import bbox_pred, clip_boxes, clip_pad
from helper.processing.nms import nms
import logging

DEBUG = False


class ProposalOperator(mx.operator.CustomOp):
    def __init__(self, feat_stride, scales, ratios, is_train=False, output_score=False):
        super(ProposalOperator, self).__init__()
        self._feat_stride = float(feat_stride)
        self._scales = np.fromstring(scales[1:-1], dtype=float, sep=',')
        self._ratios = np.fromstring(ratios[1:-1], dtype=float, sep=',').tolist()
        self._anchors = generate_anchors(base_size=self._feat_stride, scales=self._scales, ratios=self._ratios)
        self._num_anchors = self._anchors.shape[0]
        self._output_score = output_score

        if DEBUG:
            print 'feat_stride: {}'.format(self._feat_stride)
            print 'anchors:'
            print self._anchors

        if is_train:
            self.cfg_key = 'TRAIN'
        else:
            self.cfg_key = 'TEST'

    def forward(self, is_train, req, in_data, out_data, aux):
        # for each (H, W) location i
        #   generate A anchor boxes centered on cell i
        #   apply predicted bbox deltas at cell i to each of the A anchors
        # clip predicted boxes to image
        # remove predicted boxes with either height or width < threshold
        # sort all (proposal, score) pairs by score from highest to lowest
        # take top pre_nms_topN proposals before NMS
        # apply NMS with threshold 0.7 to remaining proposals
        # take after_nms_topN proposals after NMS
        # return the top proposals (-> RoIs top, scores top)
        pre_nms_topN = config[self.cfg_key].RPN_PRE_NMS_TOP_N
        post_nms_topN = config[self.cfg_key].RPN_POST_NMS_TOP_N
        nms_thresh = config[self.cfg_key].RPN_NMS_THRESH
        min_size = config[self.cfg_key].RPN_MIN_SIZE

        # the first set of anchors are background probabilities
        # keep the second part
        scores = in_data[0].asnumpy()[:, self._num_anchors:, :, :]
        if np.isnan(scores).any():
            raise ValueError("there is nan in input scores")
        bbox_deltas = in_data[1].asnumpy()
        if np.isnan(bbox_deltas).any():
            raise ValueError("there is nan in input bbox_deltas")
        im_info = in_data[2].asnumpy()[0, :]
        if DEBUG:
            print 'im_size: ({}, {})'.format(im_info[0], im_info[1])
            print 'scale: {}'.format(im_info[2])

        # 1. Generate proposals from bbox_deltas and shifted anchors
        height, width = scores.shape[-2:]
        if self.cfg_key == 'TRAIN':
            height, width = int(im_info[0] / self._feat_stride), int(im_info[1] / self._feat_stride)

        if DEBUG:
            print 'score map size: {}'.format(scores.shape)
            print "resudial = ", scores.shape[2] - height, scores.shape[3] - width
        # Enumerate all shifts
        shift_x = np.arange(0, width) * self._feat_stride
        shift_y = np.arange(0, height) * self._feat_stride
        shift_x, shift_y = np.meshgrid(shift_x, shift_y)
        shifts = np.vstack((shift_x.ravel(), shift_y.ravel(), shift_x.ravel(), shift_y.ravel())).transpose()

        # Enumerate all shifted anchors:
        #
        # add A anchors (1, A, 4) to
        # cell K shifts (K, 1, 4) to get
        # shift anchors (K, A, 4)
        # reshape to (K*A, 4) shifted anchors
        A = self._num_anchors
        K = shifts.shape[0]
        anchors = self._anchors.reshape((1, A, 4)) + shifts.reshape((1, K, 4)).transpose((1, 0, 2))
        anchors = anchors.reshape((K * A, 4))
        # Transpose and reshape predicted bbox transformations to get them
        # into the same order as the anchors:
        #
        # bbox deltas will be (1, 4 * A, H, W) format
        # transpose to (1, H, W, 4 * A)
        # reshape to (1 * H * W * A, 4) where rows are ordered by (h, w, a)
        # in slowest to fastest order
        bbox_deltas = clip_pad(bbox_deltas, (height, width))
        bbox_deltas = bbox_deltas.transpose((0, 2, 3, 1)).reshape((-1, 4))

        # Same story for the scores:
        #
        # scores are (1, A, H, W) format
        # transpose to (1, H, W, A)
        # reshape to (1 * H * W * A, 1) where rows are ordered by (h, w, a)
        scores = scores.transpose((0, 2, 3, 1)).reshape((-1, 1))

        # Convert anchors into proposals via bbox transformations
        proposals = bbox_pred(anchors, bbox_deltas)

        # 2. clip predicted boxes to image
        proposals = clip_boxes(proposals, im_info[:2])

        # 3. remove predicted boxes with either height or width < threshold
        # (NOTE: convert min_size to input image scale stored in im_info[2])
        keep = ProposalOperator._filter_boxes(proposals, min_size * im_info[2])

        proposals = proposals[keep, :]
        scores = scores[keep]
        # 4. sort all (proposal, score) pairs by score from highest to lowest
        # 5. take top pre_nms_topN (e.g. 6000)
        order = scores.ravel().argsort()[::-1]
        if pre_nms_topN > 0:
            order = order[:pre_nms_topN]
        proposals = proposals[order, :]
        scores = scores[order]
        # 6. apply nms (e.g. threshold = 0.7)
        # 7. take after_nms_topN (e.g. 300)
        # 8. return the top proposals (-> RoIs top)
        keep = nms(np.hstack((proposals, scores)), nms_thresh)
        if post_nms_topN > 0:
            keep = keep[:post_nms_topN]
        # pad to ensure output size remains unchanged
        if len(keep) < post_nms_topN:
            if len(keep) == 0:
                logging.log(logging.ERROR, "currently len(keep) is zero")
            pad = npr.choice(keep, size=post_nms_topN - len(keep))
            keep = np.hstack((keep, pad))
        proposals = proposals[keep, :]
        scores = scores[keep]
        # Output rois array
        # Our RPN implementation only supports a single input image, so all
        # batch inds are 0
        batch_inds = np.zeros((proposals.shape[0], 1), dtype=np.float32)
        blob = np.hstack((batch_inds, proposals.astype(np.float32, copy=False)))
        self.assign(out_data[0], req[0], blob)
        if self._output_score:
            self.assign(out_data[1], req[1], scores.astype(np.float32, copy=False))

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        self.assign(in_grad[0], req[0], 0)
        self.assign(in_grad[1], req[1], 0)

    @staticmethod
    def _filter_boxes(boxes, min_size):
        """ Remove all boxes with any side smaller than min_size """
        ws = boxes[:, 2] - boxes[:, 0] + 1
        hs = boxes[:, 3] - boxes[:, 1] + 1
        keep = np.where((ws >= min_size) & (hs >= min_size))[0]
        return keep


@mx.operator.register("proposal")
class ProposalProp(mx.operator.CustomOpProp):
    def __init__(self, feat_stride, scales, ratios, is_train=False, output_score=False):
        super(ProposalProp, self).__init__(need_top_grad=False)
        self._feat_stride = feat_stride
        self._scales = scales
        self._ratios = ratios
        self._is_train = True if is_train == 'True' else False
        self._output_score = output_score

        if self._is_train:
            self.cfg_key = 'TRAIN'
        else:
            self.cfg_key = 'TEST'

    def list_arguments(self):
        return ['cls_prob', 'bbox_pred', 'im_info']

    def list_outputs(self):
        if self._output_score:
            return ['output', 'score']
        else:
            return ['output']

    def infer_shape(self, in_shape):
        cfg_key = self.cfg_key
        cls_prob_shape = in_shape[0]
        bbox_pred_shape = in_shape[1]
        assert cls_prob_shape[0] == bbox_pred_shape[0], 'ROI number does not equal in cls and reg'

        batch_size = cls_prob_shape[0]
        if batch_size > 1:
            raise ValueError("Only single item batches are supported")

        im_info_shape = (batch_size, 3)
        output_shape = (config[cfg_key].RPN_POST_NMS_TOP_N, 5)
        score_shape = (config[cfg_key].RPN_POST_NMS_TOP_N, 1)

        if self._output_score:
            return [cls_prob_shape, bbox_pred_shape, im_info_shape], [output_shape, score_shape]
        else:
            return [cls_prob_shape, bbox_pred_shape, im_info_shape], [output_shape]

    def create_operator(self, ctx, shapes, dtypes):
        return ProposalOperator(self._feat_stride, self._scales, self._ratios, self._is_train, self._output_score)

    def declare_backward_dependency(self, out_grad, in_data, out_data):
        return []
