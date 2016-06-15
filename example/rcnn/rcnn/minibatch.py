"""
To construct data iterator from imdb, batch sampling procedure are defined here
training minibatch =
    {'data': [num_images, c, h, w],
    'rois': [num_rois, 5],
    'labels': [num_rois],
    'bbox_targets': [num_rois, 4 * num_classes],
    'bbox_inside_weights': [num_rois, 4 * num_classes],
    'bbox_outside_weights': [num_rois, 4 * num_classes]}
    num_images should divide config['TRAIN_BATCH_SIZE'] and num_rois = config['TRAIN_BATCH_SIZE'] / num_images
validation minibatch is similar except num_images = 1 and num_rois = all rois
testing minibatch =
    {'data': [num_images, c, h, w],
    'rois': [num_rois, 5]}
    num_images = 1 and num_rois = all rois
"""

import cv2
import numpy as np
import numpy.random as npr

from helper.processing import image_processing
from helper.processing.bbox_regression import expand_bbox_regression_targets
from rcnn.config import config


def get_minibatch(roidb, num_classes):
    """
    return minibatch of images in roidb
    :param roidb: subset of main database
    :param num_classes: number of classes is used in bbox regression targets
    :return: minibatch: {'data', 'rois', 'labels', 'bbox_targets', 'bbox_inside_weights', 'bbox_outside_weights'}
    """
    num_images = len(roidb)
    random_scale_indexes = npr.randint(0, high=len(config.TRAIN.SCALES), size=num_images)
    assert config.TRAIN.BATCH_SIZE % num_images == 0, \
        'num_images {} must devide BATCHSIZE {}'.format(num_images, config.TRAIN.BATCH_SIZE)
    rois_per_image = config.TRAIN.BATCH_SIZE / num_images
    fg_rois_per_image = np.round(config.TRAIN.FG_FRACTION * rois_per_image).astype(int)

    # im_array: [num_images, c, h, w]
    im_array, im_scales = get_image_array(roidb, config.TRAIN.SCALES, random_scale_indexes)

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

    rois_array = np.vstack(rois_array)
    labels_array = np.hstack(labels_array)
    bbox_targets_array = np.vstack(bbox_targets_array)
    bbox_inside_array = np.vstack(bbox_inside_array)
    bbox_outside_array = np.array(bbox_inside_array > 0).astype(np.float32)

    minibatch = {'data': im_array,
                 'rois': rois_array,
                 'labels': labels_array,
                 'bbox_targets': bbox_targets_array,
                 'bbox_inside_weights': bbox_inside_array,
                 'bbox_outside_weights': bbox_outside_array}
    return minibatch


def get_testbatch(roidb, num_classes):
    """
    return test batch of given roidb
    actually, there is only one testing scale and len(roidb) is 1
    :param roidb: subset of main database
    :param num_classes: number of classes is used in bbox regression targets
    :return: minibatch: {'data', 'rois'}
    """
    num_images = len(roidb)
    random_scale_indexes = npr.randint(0, high=len(config.TEST.SCALES), size=num_images)
    im_array, im_scales = get_image_array(roidb, config.TEST.SCALES, random_scale_indexes)

    rois_array = list()
    for im_i in range(num_images):
        im_rois = roidb[im_i]['boxes']
        rois = im_rois * im_scales[im_i]
        batch_index = im_i * np.ones((rois.shape[0], 1))
        rois_array_this_image = np.hstack((batch_index, rois))
        rois_array.append(rois_array_this_image)

    rois_array = np.vstack(rois_array)

    testbatch = {'data': im_array,
                 'rois': rois_array}
    return testbatch


def get_image_array(roidb, scales, scale_indexes):
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
        im, im_scale = image_processing.resize(im, target_size, config.TRAIN.MAX_SIZE)
        im_tensor = image_processing.transform(im, config.PIXEL_MEANS)
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
