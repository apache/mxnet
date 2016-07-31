import os

from helper.dataset import pascal_voc
from helper.processing import roidb
from rcnn import data_iter

# test flip
devkit_path = os.path.join(os.path.expanduser('~'), 'Dataset', 'VOCdevkit')
voc = pascal_voc.PascalVOC('trainval', '2007', devkit_path)
gt_roidb = voc.gt_roidb()
ss_roidb = voc.selective_search_roidb(gt_roidb)
ss_roidb = voc.append_flipped_images(ss_roidb)
roidb.prepare_roidb(voc, ss_roidb)
means, stds = roidb.add_bbox_regression_targets(ss_roidb)

roi_iter = data_iter.ROIIter(ss_roidb, shuffle=True)

for j in range(0, 20):
    print j
    for databatch in roi_iter:
        i = 0
    roi_iter.reset()
