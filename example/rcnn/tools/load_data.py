from helper.dataset.pascal_voc import PascalVOC
from helper.processing.roidb import prepare_roidb, add_bbox_regression_targets


def load_train_roidb(image_set, year, root_path, devkit_path, flip=False):
    voc = PascalVOC(image_set, year, root_path, devkit_path)
    gt_roidb = voc.gt_roidb()
    ss_roidb = voc.selective_search_roidb(gt_roidb)
    if flip:
        ss_roidb = voc.append_flipped_images(ss_roidb)
    prepare_roidb(voc, ss_roidb)
    means, stds = add_bbox_regression_targets(ss_roidb)
    return voc, ss_roidb, means, stds


def load_test_roidb(image_set, year, root_path, devkit_path):
    voc = PascalVOC(image_set, year, root_path, devkit_path)
    gt_roidb = voc.gt_roidb()
    ss_roidb = voc.selective_search_roidb(gt_roidb)
    prepare_roidb(voc, ss_roidb)
    return voc, ss_roidb
