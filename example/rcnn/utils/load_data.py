from helper.dataset.pascal_voc import PascalVOC
from helper.processing.roidb import prepare_roidb, add_bbox_regression_targets


def load_ss_roidb(image_set, year, root_path, devkit_path, flip=False):
    voc = PascalVOC(image_set, year, root_path, devkit_path)
    gt_roidb = voc.gt_roidb()
    ss_roidb = voc.selective_search_roidb(gt_roidb)
    if flip:
        ss_roidb = voc.append_flipped_images(ss_roidb)
    prepare_roidb(voc, ss_roidb)
    means, stds = add_bbox_regression_targets(ss_roidb)
    return voc, ss_roidb, means, stds


def load_gt_roidb(image_set, year, root_path, devkit_path, flip=False):
    voc = PascalVOC(image_set, year, root_path, devkit_path)
    gt_roidb = voc.gt_roidb()
    if flip:
        gt_roidb = voc.append_flipped_images(gt_roidb)
    prepare_roidb(voc, gt_roidb)
    return voc, gt_roidb


def load_rpn_roidb(image_set, year, root_path, devkit_path, flip=False):
    voc = PascalVOC(image_set, year, root_path, devkit_path)
    gt_roidb = voc.gt_roidb()
    rpn_roidb = voc.rpn_roidb(gt_roidb)
    if flip:
        rpn_roidb = voc.append_flipped_images(rpn_roidb)
    prepare_roidb(voc, rpn_roidb)
    means, stds = add_bbox_regression_targets(rpn_roidb)
    return voc, rpn_roidb, means, stds


def load_test_ss_roidb(image_set, year, root_path, devkit_path):
    voc = PascalVOC(image_set, year, root_path, devkit_path)
    gt_roidb = voc.gt_roidb()
    ss_roidb = voc.selective_search_roidb(gt_roidb)
    prepare_roidb(voc, ss_roidb)
    return voc, ss_roidb


def load_test_rpn_roidb(image_set, year, root_path, devkit_path):
    voc = PascalVOC(image_set, year, root_path, devkit_path)
    gt_roidb = voc.gt_roidb()
    rpn_roidb = voc.rpn_roidb(gt_roidb)
    prepare_roidb(voc, rpn_roidb)
    return voc, rpn_roidb
