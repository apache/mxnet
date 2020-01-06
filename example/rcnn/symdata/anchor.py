# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

import numpy as np
from symdata.bbox import bbox_overlaps, bbox_transform


class AnchorGenerator:
    def __init__(self, feat_stride=16, anchor_scales=(8, 16, 32), anchor_ratios=(0.5, 1, 2)):
        self._num_anchors = len(anchor_scales) * len(anchor_ratios)
        self._feat_stride = feat_stride
        self._base_anchors = self._generate_base_anchors(feat_stride, np.array(anchor_scales), np.array(anchor_ratios))

    def generate(self, feat_height, feat_width):
        shift_x = np.arange(0, feat_width) * self._feat_stride
        shift_y = np.arange(0, feat_height) * self._feat_stride
        shift_x, shift_y = np.meshgrid(shift_x, shift_y)
        shifts = np.vstack((shift_x.ravel(), shift_y.ravel(), shift_x.ravel(), shift_y.ravel())).transpose()
        # add A anchors (1, A, 4) to
        # cell K shifts (K, 1, 4) to get
        # shift anchors (K, A, 4)
        # reshape to (K*A, 4) shifted anchors
        A = self._num_anchors
        K = shifts.shape[0]
        all_anchors = self._base_anchors.reshape((1, A, 4)) + shifts.reshape((1, K, 4)).transpose((1, 0, 2))
        all_anchors = all_anchors.reshape((K * A, 4))
        return all_anchors

    @staticmethod
    def _generate_base_anchors(base_size, scales, ratios):
        """
        Generate anchor (reference) windows by enumerating aspect ratios X
        scales wrt a reference (0, 0, 15, 15) window.
        """
        base_anchor = np.array([1, 1, base_size, base_size]) - 1
        ratio_anchors = AnchorGenerator._ratio_enum(base_anchor, ratios)
        anchors = np.vstack([AnchorGenerator._scale_enum(ratio_anchors[i, :], scales)
                             for i in range(ratio_anchors.shape[0])])
        return anchors

    @staticmethod
    def _whctrs(anchor):
        """
        Return width, height, x center, and y center for an anchor (window).
        """
        w = anchor[2] - anchor[0] + 1
        h = anchor[3] - anchor[1] + 1
        x_ctr = anchor[0] + 0.5 * (w - 1)
        y_ctr = anchor[1] + 0.5 * (h - 1)
        return w, h, x_ctr, y_ctr

    @staticmethod
    def _mkanchors(ws, hs, x_ctr, y_ctr):
        """
        Given a vector of widths (ws) and heights (hs) around a center
        (x_ctr, y_ctr), output a set of anchors (windows).
        """
        ws = ws[:, np.newaxis]
        hs = hs[:, np.newaxis]
        anchors = np.hstack((x_ctr - 0.5 * (ws - 1),
                             y_ctr - 0.5 * (hs - 1),
                             x_ctr + 0.5 * (ws - 1),
                             y_ctr + 0.5 * (hs - 1)))
        return anchors

    @staticmethod
    def _ratio_enum(anchor, ratios):
        """
        Enumerate a set of anchors for each aspect ratio wrt an anchor.
        """
        w, h, x_ctr, y_ctr = AnchorGenerator._whctrs(anchor)
        size = w * h
        size_ratios = size / ratios
        ws = np.round(np.sqrt(size_ratios))
        hs = np.round(ws * ratios)
        anchors = AnchorGenerator._mkanchors(ws, hs, x_ctr, y_ctr)
        return anchors

    @staticmethod
    def _scale_enum(anchor, scales):
        """
        Enumerate a set of anchors for each scale wrt an anchor.
        """
        w, h, x_ctr, y_ctr = AnchorGenerator._whctrs(anchor)
        ws = w * scales
        hs = h * scales
        anchors = AnchorGenerator._mkanchors(ws, hs, x_ctr, y_ctr)
        return anchors


class AnchorSampler:
    def __init__(self, allowed_border=0, batch_rois=256, fg_fraction=0.5, fg_overlap=0.7, bg_overlap=0.3):
        self._allowed_border = allowed_border
        self._num_batch = batch_rois
        self._num_fg = int(batch_rois * fg_fraction)
        self._fg_overlap = fg_overlap
        self._bg_overlap = bg_overlap

    def assign(self, anchors, gt_boxes, im_height, im_width):
        num_anchors = anchors.shape[0]

        # filter out padded gt_boxes
        valid_labels = np.where(gt_boxes[:, -1] > 0)[0]
        gt_boxes = gt_boxes[valid_labels]

        # filter out anchors outside the region
        inds_inside = np.where((anchors[:, 0] >= -self._allowed_border) &
                               (anchors[:, 2] < im_width + self._allowed_border) &
                               (anchors[:, 1] >= -self._allowed_border) &
                               (anchors[:, 3] < im_height + self._allowed_border))[0]
        anchors = anchors[inds_inside, :]
        num_valid = len(inds_inside)

        # label: 1 is positive, 0 is negative, -1 is dont care
        labels = np.ones((num_valid,), dtype=np.float32) * -1
        bbox_targets = np.zeros((num_valid, 4), dtype=np.float32)
        bbox_weights = np.zeros((num_valid, 4), dtype=np.float32)

        # sample for positive labels
        if gt_boxes.size > 0:
            # overlap between the anchors and the gt boxes
            # overlaps (ex, gt)
            overlaps = bbox_overlaps(anchors.astype(np.float), gt_boxes.astype(np.float))
            gt_max_overlaps = overlaps.max(axis=0)

            # fg anchors: anchor with highest overlap for each gt; or overlap > iou thresh
            fg_inds = np.where((overlaps >= self._fg_overlap) | (overlaps == gt_max_overlaps))[0]

            # subsample to num_fg
            if len(fg_inds) > self._num_fg:
                fg_inds = np.random.choice(fg_inds, size=self._num_fg, replace=False)

            # bg anchor: anchor with overlap < iou thresh but not highest overlap for some gt
            bg_inds = np.where((overlaps < self._bg_overlap) & (overlaps < gt_max_overlaps))[0]

            if len(bg_inds) > self._num_batch - len(fg_inds):
                bg_inds = np.random.choice(bg_inds, size=self._num_batch - len(fg_inds), replace=False)

            # assign label
            labels[fg_inds] = 1
            labels[bg_inds] = 0

            # assign to argmax overlap
            argmax_overlaps = overlaps.argmax(axis=1)
            bbox_targets[fg_inds, :] = bbox_transform(anchors[fg_inds, :], gt_boxes[argmax_overlaps[fg_inds], :],
                                                      box_stds=(1.0, 1.0, 1.0, 1.0))

            # only fg anchors has bbox_targets
            bbox_weights[fg_inds, :] = 1
        else:
            # randomly draw bg anchors
            bg_inds = np.random.choice(np.arange(num_valid), size=self._num_batch, replace=False)
            labels[bg_inds] = 0

        all_labels = np.ones((num_anchors,), dtype=np.float32) * -1
        all_labels[inds_inside] = labels
        all_bbox_targets = np.zeros((num_anchors, 4), dtype=np.float32)
        all_bbox_targets[inds_inside, :] = bbox_targets
        all_bbox_weights = np.zeros((num_anchors, 4), dtype=np.float32)
        all_bbox_weights[inds_inside, :] = bbox_weights

        return all_labels, all_bbox_targets, all_bbox_weights
