"""
To construct data iterator from imdb, batch sampling procedure are defined here
RPN:
data =
    {'data': [num_images, c, h, w],
    'im_info': [num_images, 4] (optional)}
label =
prototype: {'gt_boxes': [num_boxes, 5]}
final:  {'label': [batch_size, 1] <- [batch_size, num_anchors, feat_height, feat_width],
         'bbox_target': [batch_size, num_anchors, feat_height, feat_width],
         'bbox_inside_weight': [batch_size, num_anchors, feat_height, feat_width],
         'bbox_outside_weight': [batch_size, num_anchors, feat_height, feat_width]}
Fast R-CNN:
data =
    {'data': [num_images, c, h, w],
    'rois': [num_images, num_rois, 5]}
label =
    {'label': [num_images, num_rois],
    'bbox_target': [num_images, num_rois, 4 * num_classes],
    'bbox_inside_weight': [num_images, num_rois, 4 * num_classes],
    'bbox_outside_weight': [num_images, num_rois, 4 * num_classes]}
"""

import cv2
import numpy as np
import numpy.random as npr

from helper.processing import image_processing
from helper.processing.bbox_regression import expand_bbox_regression_targets
from helper.processing.generate_anchor import generate_anchors
from helper.processing.bbox_regression import bbox_overlaps
from helper.processing.bbox_transform import bbox_transform
from rcnn.config import config


def get_minibatch(roidb, num_classes, mode='test', need_mean=True):
    """
    return minibatch of images in roidb
    :param roidb: a list of dict, whose length controls batch size
    :param num_classes: number of classes is used in bbox regression targets
    :param mode: controls whether blank label are returned
    :return: data, label
    """
    # build im_array: [num_images, c, h, w]
    num_images = len(roidb)
    random_scale_indexes = npr.randint(0, high=len(config.SCALES), size=num_images)
    im_array, im_scales = get_image_array(roidb, config.SCALES, random_scale_indexes, need_mean=need_mean)

    if mode == 'train':
        cfg_key = 'TRAIN'
    else:
        cfg_key = 'TEST'

    if config[cfg_key].HAS_RPN:
        assert len(roidb) == 1, 'Single batch only'
        assert len(im_scales) == 1, 'Single batch only'
        im_info = np.array([[im_array.shape[2], im_array.shape[3], im_scales[0]]], dtype=np.float32)

        data = {'data': im_array,
                'im_info': im_info}
        label = {}

        if mode == 'train':
            # gt boxes: (x1, y1, x2, y2, cls)
            gt_inds = np.where(roidb[0]['gt_classes'] != 0)[0]
            gt_boxes = np.empty((roidb[0]['boxes'].shape[0], 5), dtype=np.float32)
            gt_boxes[:, 0:4] = roidb[0]['boxes'][gt_inds, :] * im_scales[0]
            gt_boxes[:, 4] = roidb[0]['gt_classes'][gt_inds]
            label = {'gt_boxes': gt_boxes}
    else:
        if mode == 'train':
            assert config.TRAIN.BATCH_SIZE % config.TRAIN.BATCH_IMAGES == 0, \
                'BATCHIMAGES {} must devide BATCHSIZE {}'.format(config.TRAIN.BATCH_IMAGES, config.TRAIN.BATCH_SIZE)
            rois_per_image = config.TRAIN.BATCH_SIZE / config.TRAIN.BATCH_IMAGES
            fg_rois_per_image = np.round(config.TRAIN.FG_FRACTION * rois_per_image).astype(int)

            rois_array = list()
            labels_array = list()
            bbox_targets_array = list()
            bbox_inside_array = list()

            for im_i in range(num_images):
                im_rois, labels, bbox_targets, bbox_inside_weights, overlaps = \
                    sample_rois(roidb[im_i], fg_rois_per_image, rois_per_image, num_classes)

                # project im_rois
                # do not round roi
                rois = im_rois * im_scales[im_i]
                batch_index = im_i * np.ones((rois.shape[0], 1))
                rois_array_this_image = np.hstack((batch_index, rois))
                rois_array.append(rois_array_this_image)

                # add labels
                labels_array.append(labels)
                bbox_targets_array.append(bbox_targets)
                bbox_inside_array.append(bbox_inside_weights)

            rois_array = np.array(rois_array)
            labels_array = np.array(labels_array)
            bbox_targets_array = np.array(bbox_targets_array)
            bbox_inside_array = np.array(bbox_inside_array)
            bbox_outside_array = np.array(bbox_inside_array > 0).astype(np.float32)

            data = {'data': im_array,
                    'rois': rois_array}
            label = {'label': labels_array,
                     'bbox_target': bbox_targets_array,
                     'bbox_inside_weight': bbox_inside_array,
                     'bbox_outside_weight': bbox_outside_array}
        else:
            rois_array = list()
            for im_i in range(num_images):
                im_rois = roidb[im_i]['boxes']
                rois = im_rois * im_scales[im_i]
                batch_index = im_i * np.ones((rois.shape[0], 1))
                rois_array_this_image = np.hstack((batch_index, rois))
                rois_array.append(rois_array_this_image)
            rois_array = np.vstack(rois_array)

            data = {'data': im_array,
                    'rois': rois_array}
            label = {}

    return data, label


def get_image_array(roidb, scales, scale_indexes, need_mean=True):
    """
    build image array from specific roidb
    :param roidb: images to be processed
    :param scales: scale list
    :param scale_indexes: indexes
    :return: array [b, c, h, w], list of scales
    """
    num_images = len(roidb)
    processed_ims = []
    im_scales = []
    for i in range(num_images):
        im = cv2.imread(roidb[i]['image'])
        if roidb[i]['flipped']:
            im = im[:, ::-1, :]
        target_size = scales[scale_indexes[i]]
        im, im_scale = image_processing.resize(im, target_size, config.MAX_SIZE)
        im_tensor = image_processing.transform(im, config.PIXEL_MEANS, need_mean=need_mean)
        processed_ims.append(im_tensor)
        im_scales.append(im_scale)
    array = image_processing.tensor_vstack(processed_ims)
    return array, im_scales


def sample_rois(roidb, fg_rois_per_image, rois_per_image, num_classes):
    """
    generate random sample of ROIs comprising foreground and background examples
    :param roidb: database of selected rois
    :param fg_rois_per_image: foreground roi number
    :param rois_per_image: total roi number
    :param num_classes: number of classes
    :return: (labels, rois, bbox_targets, bbox_inside_weights, overlaps)
    """
    # label = class RoI has max overlap with
    labels = roidb['max_classes']
    overlaps = roidb['max_overlaps']
    rois = roidb['boxes']

    # foreground RoI with FG_THRESH overlap
    fg_indexes = np.where(overlaps >= config.TRAIN.FG_THRESH)[0]
    # guard against the case when an image has fewer than fg_rois_per_image foreground RoIs
    fg_rois_per_this_image = np.minimum(fg_rois_per_image, fg_indexes.size)
    # Sample foreground regions without replacement
    if fg_indexes.size > 0:
        fg_indexes = npr.choice(fg_indexes, size=fg_rois_per_this_image, replace=False)

    # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
    bg_indexes = np.where((overlaps < config.TRAIN.BG_THRESH_HI) & (overlaps >= config.TRAIN.BG_THRESH_LO))[0]
    # Compute number of background RoIs to take from this image (guarding against there being fewer than desired)
    bg_rois_per_this_image = rois_per_image - fg_rois_per_this_image
    bg_rois_per_this_image = np.minimum(bg_rois_per_this_image, bg_indexes.size)
    # Sample foreground regions without replacement
    if bg_indexes.size > 0:
        bg_indexes = npr.choice(bg_indexes, size=bg_rois_per_this_image, replace=False)

    # indexes selected
    keep_indexes = np.append(fg_indexes, bg_indexes)

    # pad more to ensure a fixed minibatch size
    if keep_indexes.shape[0] < rois_per_image:
        gap = rois_per_image - keep_indexes.shape[0]
        gap_indexes = npr.choice(range(len(rois)), size=gap, replace=False)
        keep_indexes = np.append(keep_indexes, gap_indexes)

    # select labels
    labels = labels[keep_indexes]
    # set labels of bg_rois to be 0
    labels[fg_rois_per_this_image:] = 0
    overlaps = overlaps[keep_indexes]
    rois = rois[keep_indexes]

    bbox_targets, bbox_inside_weights = \
        expand_bbox_regression_targets(roidb['bbox_targets'][keep_indexes, :], num_classes)

    return rois, labels, bbox_targets, bbox_inside_weights, overlaps


def assign_anchor(feat_shape, gt_boxes, im_info, feat_stride=16,
                  scales=(8, 16, 32), ratios=(0.5, 1, 2), allowed_border=0):
    """
    assign ground truth boxes to anchor positions
    :param feat_shape: infer output shape
    :param gt_boxes: assign ground truth
    :param im_info: filter out anchors overlapped with edges
    :param feat_stride: anchor position step
    :param scales: used to generate anchors, affects num_anchors (per location)
    :param ratios: aspect ratios of generated anchors
    :param allowed_border: filter out anchors with edge overlap > allowed_border
    :return: dict of label
    'label': of shape (batch_size, 1) <- (batch_size, num_anchors, feat_height, feat_width)
    'bbox_target': of shape (batch_size, num_anchors * 4, feat_height, feat_width)
    'bbox_inside_weight': *todo* mark the assigned anchors
    'bbox_outside_weight': used to normalize the bbox_loss, all weights sums to RPN_POSITIVE_WEIGHT
    """
    def _unmap(data, count, inds, fill=0):
        """" unmap a subset inds of data into original data of size count """
        if len(data.shape) == 1:
            ret = np.empty((count,), dtype=np.float32)
            ret.fill(fill)
            ret[inds] = data
        else:
            ret = np.empty((count,) + data.shape[1:], dtype=np.float32)
            ret.fill(fill)
            ret[inds, :] = data
        return ret

    def _compute_targets(ex_rois, gt_rois):
        """ compute bbox targets for an image """
        assert ex_rois.shape[0] == gt_rois.shape[0]
        assert ex_rois.shape[1] == 4
        assert gt_rois.shape[1] == 5

        return bbox_transform(ex_rois, gt_rois[:, :4]).astype(np.float32, copy=False)

    DEBUG = False
    im_info = im_info[0]
    scales = np.array(scales, dtype=np.float32)
    base_anchors = generate_anchors(base_size=16, ratios=list(ratios), scales=scales)
    num_anchors = base_anchors.shape[0]
    feat_height, feat_width = feat_shape[-2:]

    if DEBUG:
        print 'anchors:'
        print base_anchors
        print 'anchor shapes:'
        print np.hstack((base_anchors[:, 2::4] - base_anchors[:, 0::4],
                         base_anchors[:, 3::4] - base_anchors[:, 1::4]))
        print 'im_info', im_info
        print 'height', feat_height, 'width', feat_width
        print 'gt_boxes shape', gt_boxes.shape
        print 'gt_boxes', gt_boxes

    # 1. generate proposals from bbox deltas and shifted anchors
    shift_x = np.arange(0, feat_width) * feat_stride
    shift_y = np.arange(0, feat_height) * feat_stride
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    shifts = np.vstack((shift_x.ravel(), shift_y.ravel(), shift_x.ravel(), shift_y.ravel())).transpose()
    # add A anchors (1, A, 4) to
    # cell K shifts (K, 1, 4) to get
    # shift anchors (K, A, 4)
    # reshape to (K*A, 4) shifted anchors
    A = num_anchors
    K = shifts.shape[0]
    all_anchors = base_anchors.reshape((1, A, 4)) + shifts.reshape((1, K, 4)).transpose((1, 0, 2))
    all_anchors = all_anchors.reshape((K * A, 4))
    total_anchors = int(K * A)

    # only keep anchors inside the image
    inds_inside = np.where((all_anchors[:, 0] >= -allowed_border) &
                           (all_anchors[:, 1] >= -allowed_border) &
                           (all_anchors[:, 2] < im_info[1] + allowed_border) &
                           (all_anchors[:, 3] < im_info[0] + allowed_border))[0]
    if DEBUG:
        print 'total_anchors', total_anchors
        print 'inds_inside', len(inds_inside)

    # keep only inside anchors
    anchors = all_anchors[inds_inside, :]
    if DEBUG:
        print 'anchors shape', anchors.shape

    # label: 1 is positive, 0 is negative, -1 is dont care
    labels = np.empty((len(inds_inside),), dtype=np.float32)
    labels.fill(-1)

    if gt_boxes.size > 0:
        # overlap between the anchors and the gt boxes
        # overlaps (ex, gt)
        overlaps = bbox_overlaps(anchors.astype(np.float), gt_boxes.astype(np.float))
        argmax_overlaps = overlaps.argmax(axis=1)
        max_overlaps = overlaps[np.arange(len(inds_inside)), argmax_overlaps]
        gt_argmax_overlaps = overlaps.argmax(axis=0)
        gt_max_overlaps = overlaps[gt_argmax_overlaps, np.arange(overlaps.shape[1])]
        gt_argmax_overlaps = np.where(overlaps == gt_max_overlaps)[0]

        if not config.TRAIN.RPN_CLOBBER_POSITIVES:
            # assign bg labels first so that positive labels can clobber them
            labels[max_overlaps < config.TRAIN.RPN_NEGATIVE_OVERLAP] = 0

        # fg label: for each gt, anchor with highest overlap
        labels[gt_argmax_overlaps] = 1

        # fg label: above threshold IoU
        labels[max_overlaps >= config.TRAIN.RPN_POSITIVE_OVERLAP] = 1

        if config.TRAIN.RPN_CLOBBER_POSITIVES:
            # assign bg labels last so that negative labels can clobber positives
            labels[max_overlaps < config.TRAIN.RPN_NEGATIVE_OVERLAP] = 0
    else:
        labels[:] = 0

    # subsample positive labels if we have too many
    num_fg = int(config.TRAIN.RPN_FG_FRACTION * config.TRAIN.RPN_BATCH_SIZE)
    fg_inds = np.where(labels == 1)[0]
    if len(fg_inds) > num_fg:
        disable_inds = npr.choice(fg_inds, size=(len(fg_inds) - num_fg), replace=False)
        if DEBUG:
            disable_inds = fg_inds[:(len(fg_inds) - num_fg)]
        labels[disable_inds] = -1

    # subsample negative labels if we have too many
    num_bg = config.TRAIN.RPN_BATCH_SIZE - np.sum(labels == 1)
    bg_inds = np.where(labels == 0)[0]
    if len(bg_inds) > num_bg:
        disable_inds = npr.choice(bg_inds, size=(len(bg_inds) - num_bg), replace=False)
        if DEBUG:
            disable_inds = bg_inds[:(len(bg_inds) - num_bg)]
        labels[disable_inds] = -1

    bbox_targets = np.zeros((len(inds_inside), 4), dtype=np.float32)
    if gt_boxes.size > 0:
        bbox_targets[:] = _compute_targets(anchors, gt_boxes[argmax_overlaps, :])

    bbox_inside_weights = np.zeros((len(inds_inside), 4), dtype=np.float32)
    bbox_inside_weights[labels == 1, :] = np.array(config.TRAIN.RPN_BBOX_INSIDE_WEIGHTS)

    bbox_outside_weights = np.zeros((len(inds_inside), 4), dtype=np.float32)
    if config.TRAIN.RPN_POSITIVE_WEIGHT < 0:
        # uniform weighting of exampling (given non-uniform sampling)
        num_examples = np.sum(labels >= 0)
        positive_weights = np.ones((1, 4)) * 1.0 / num_examples
        negative_weights = np.ones((1, 4)) * 1.0 / num_examples
    else:
        assert ((config.TRAIN.RPN_POSITIVE_WEIGHT > 0) & (config.TRAIN.RPN_POSITIVE_WEIGHT < 1))
        positive_weights = config.TRAIN.RPN_POSITIVE_WEIGHT / np.sum(labels == 1)
        negative_weights = (1.0 - config.TRAIN.RPN_POSITIVE_WEIGHT) / np.sum(labels == 1)
    bbox_outside_weights[labels == 1, :] = positive_weights
    bbox_outside_weights[labels == 0, :] = negative_weights

    if DEBUG:
        _sums = bbox_targets[labels == 1, :].sum(axis=0)
        _squared_sums = (bbox_targets[labels == 1, :] ** 2).sum(axis=0)
        _counts = config.EPS + np.sum(labels == 1)
        means = _sums / _counts
        stds = np.sqrt(_squared_sums / _counts - means ** 2)
        print 'means', means
        print 'stdevs', stds

    # map up to original set of anchors
    labels = _unmap(labels, total_anchors, inds_inside, fill=-1)
    bbox_targets = _unmap(bbox_targets, total_anchors, inds_inside, fill=0)
    bbox_inside_weights = _unmap(bbox_inside_weights, total_anchors, inds_inside, fill=0)
    bbox_outside_weights = _unmap(bbox_outside_weights, total_anchors, inds_inside, fill=0)

    if DEBUG:
        print 'rpn: max max_overlaps', np.max(max_overlaps)
        print 'rpn: num_positives', np.sum(labels == 1)
        print 'rpn: num_negatives', np.sum(labels == 0)
        _fg_sum = np.sum(labels == 1)
        _bg_sum = np.sum(labels == 0)
        _count = 1
        print 'rpn: num_positive avg', _fg_sum / _count
        print 'rpn: num_negative avg', _bg_sum / _count

    labels = labels.reshape((1, feat_height, feat_width, A)).transpose(0, 3, 1, 2)
    labels = labels.reshape((1, A * feat_height * feat_width))
    bbox_targets = bbox_targets.reshape((1, feat_height, feat_width, A * 4)).transpose(0, 3, 1, 2)
    bbox_inside_weights = bbox_inside_weights.reshape((1, feat_height, feat_width, A * 4)).transpose((0, 3, 1, 2))
    bbox_outside_weights = bbox_outside_weights.reshape((1, feat_height, feat_width, A * 4)).transpose((0, 3, 1, 2))

    label = {'label': labels,
             'bbox_target': bbox_targets,
             'bbox_inside_weight': bbox_inside_weights,
             'bbox_outside_weight': bbox_outside_weights}

    if config.END2END == 1:
        label.update({'gt_boxes': gt_boxes})

    return label
