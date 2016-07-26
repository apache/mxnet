"""
General image database
An image database creates a list of relative image path called image_set_index and
transform index to absolute image path. As to training, it is necessary that ground
truth and proposals are mixed together for training.
"""

import numpy as np
import scipy.sparse
from helper.processing.bbox_regression import bbox_overlaps


class IMDB(object):
    def __init__(self, name):
        self.name = name
        self.classes = []
        self.num_classes = 0
        self.image_set_index = []
        self.num_images = 0

        self.config = {}

    def image_path_from_index(self, index):
        raise NotImplementedError

    def gt_roidb(self):
        raise NotImplementedError

    def roidb(self, gt_roidb):
        raise NotImplementedError

    def create_roidb_from_box_list(self, box_list, gt_roidb):
        """
        given ground truth, prepare roidb
        :param box_list: [image_index] ndarray of [box_index][x1, x2, y1, y2]
        :param gt_roidb: [image_index]['boxes', 'gt_classes', 'gt_overlaps', 'flipped']
        :return: roidb: [image_index]['boxes', 'gt_classes', 'gt_overlaps', 'flipped']
        """
        assert len(box_list) == self.num_images, 'number of boxes matrix must match number of images'
        roidb = []
        for i in range(self.num_images):
            boxes = box_list[i]
            num_boxes = boxes.shape[0]
            overlaps = np.zeros((num_boxes, self.num_classes), dtype=np.float32)

            if gt_roidb is not None and gt_roidb[i]['boxes'].size > 0:
                gt_boxes = gt_roidb[i]['boxes']
                gt_classes = gt_roidb[i]['gt_classes']
                # n boxes and k gt_boxes => n * k overlap
                gt_overlaps = bbox_overlaps(boxes.astype(np.float), gt_boxes.astype(np.float))

                # for each box in n boxes, select only maximum overlap (must be greater than zero)
                argmaxes = gt_overlaps.argmax(axis=1)
                maxes = gt_overlaps.max(axis=1)
                I = np.where(maxes > 0)[0]
                overlaps[I, gt_classes[argmaxes[I]]] = maxes[I]

            overlaps = scipy.sparse.csr_matrix(overlaps)
            roidb.append({'boxes': boxes,
                          'gt_classes': np.zeros((num_boxes,), dtype=np.int32),
                          'gt_overlaps': overlaps,
                          'flipped': False})

        return roidb

    @staticmethod
    def merge_roidbs(a, b):
        """
        merge roidbs into one
        :param a: roidb to be merged into
        :param b: roidb to be merged
        :return: merged imdb
        """
        assert len(a) == len(b)
        for i in range(len(a)):
            a[i]['boxes'] = np.vstack((a[i]['boxes'], b[i]['boxes']))
            a[i]['gt_classes'] = np.hstack((a[i]['gt_classes'], b[i]['gt_classes']))
            a[i]['gt_overlaps'] = scipy.sparse.vstack([a[i]['gt_overlaps'], b[i]['gt_overlaps']])
        return a

    def append_flipped_images(self, roidb):
        """
        append flipped images to an roidb
        flip boxes coordinates, images will be actually flipped when loading into network
        :param roidb: [image_index]['boxes', 'gt_classes', 'gt_overlaps', 'flipped']
        :return: roidb: [image_index]['boxes', 'gt_classes', 'gt_overlaps', 'flipped']
        """
        print 'append flipped images to roidb'
        import cv2
        assert self.num_images == len(roidb)
        widths = [cv2.imread(self.image_path_from_index(self.image_set_index[i])).shape[1]
                  for i in range(self.num_images)]
        for i in range(self.num_images):
            boxes = roidb[i]['boxes'].copy()
            oldx1 = boxes[:, 0].copy()
            oldx2 = boxes[:, 2].copy()
            boxes[:, 0] = widths[i] - oldx2 - 1
            boxes[:, 2] = widths[i] - oldx1 - 1
            assert (boxes[:, 2] >= boxes[:, 0]).all()
            entry = {'boxes': boxes,
                     'gt_classes': roidb[i]['gt_classes'],
                     'gt_overlaps': roidb[i]['gt_overlaps'],
                     'flipped': True}
            roidb.append(entry)

        self.image_set_index *= 2
        return roidb

    def evaluate_recall(self, roidb, candidate_boxes=None, thresholds=None, area='all', limit=None):
        """
        evaluate detection proposal recall metrics
        record max overlap value for each gt box; return vector of overlap values
        :param roidb: used to evaluate
        :param candidate_boxes: if not given, use roidb's non-gt boxes
        :param thresholds: array-like recall threshold
        :param area: index in area ranges
        :param limit: limit of bounding box evaluated
        :return: None
        ar: average recall, recalls: vector recalls at each IoU overlap threshold
        thresholds: vector of IoU overlap threshold, gt_overlaps: vector of all ground-truth overlaps
        """
        areas = {'all': 0, 'small': 1, 'medium': 2, 'large': 3,
                 '96-128': 4, '128-256': 5, '256-512': 6, '512-inf': 7}
        area_ranges = [[0**2, 1e5**2], [0**2, 32**2], [32**2, 96**2], [96**2, 1e5**2],
                       [96**2, 128**2], [128**2, 256**2], [256**2, 512**2], [512**2, 1e5**2]]
        assert areas.has_key(area), 'unknown area range: {}'.format(area)
        area_range = area_ranges[areas[area]]
        gt_overlaps = np.zeros(0)
        num_pos = 0
        for i in range(self.num_images):
            # check for max_overlaps == 1 avoids including crowd annotations
            max_gt_overlaps = roidb[i]['gt_overlaps'].toarray().max(axis=1)
            gt_inds = np.where((roidb[i]['gt_classes'] > 0) & (max_gt_overlaps == 1))[0]
            gt_boxes = roidb[i]['boxes'][gt_inds, :]
            gt_areas = (gt_boxes[:, 2] - gt_boxes[:, 0] + 1) * (gt_boxes[:, 3] - gt_boxes[:, 1] + 1)
            valid_gt_inds = np.where((gt_areas >= area_range[0]) & (gt_areas <= area_range[1]))[0]
            gt_boxes = gt_boxes[valid_gt_inds, :]
            num_pos += len(valid_gt_inds)

            if candidate_boxes is None:
                # default is use the non-gt boxes from roidb
                non_gt_inds = np.where(roidb[i]['gt_classes'] == 0)[0]
                boxes = roidb[i]['boxes'][non_gt_inds, :]
            else:
                boxes = candidate_boxes[i]
            if boxes.shape[0] == 0:
                continue
            if limit is not None and boxes.shape[0] > limit:
                boxes = boxes[:limit, :]

            overlaps = bbox_overlaps(boxes.astype(np.float), gt_boxes.astype(np.float))

            _gt_overlaps = np.zeros((gt_boxes.shape[0]))
            for j in range(gt_boxes.shape[0]):
                # find which proposal maximally covers each gt box
                argmax_overlaps = overlaps.argmax(axis=0)
                # get the IoU amount of coverage for each gt box
                max_overlaps = overlaps.max(axis=0)
                # find which gt box is covered by most IoU
                gt_ind = max_overlaps.argmax()
                gt_ovr = max_overlaps.max()
                assert (gt_ovr >= 0)
                # find the proposal box that covers the best covered gt box
                box_ind = argmax_overlaps[gt_ind]
                # record the IoU coverage of this gt box
                _gt_overlaps[j] = overlaps[box_ind, gt_ind]
                assert (_gt_overlaps[j] == gt_ovr)
                # mark the proposal box and the gt box as used
                overlaps[box_ind, :] = -1
                overlaps[:, gt_ind] = -1
            # append recorded IoU coverage level
            gt_overlaps = np.hstack((gt_overlaps, _gt_overlaps))

        gt_overlaps = np.sort(gt_overlaps)
        if thresholds is None:
            step = 0.05
            thresholds = np.arange(0.5, 0.95 + 1e-5, step)
        recalls = np.zeros_like(thresholds)

        # compute recall for each IoU threshold
        for i, t in enumerate(thresholds):
            recalls[i] = (gt_overlaps >= t).sum() / float(num_pos)
        ar = recalls.mean()

        # print results
        print 'average recall: {:.3f}'.format(ar)
        for threshold, recall in zip(thresholds, recalls):
            print 'recall @{:.2f}: {:.3f}'.format(threshold, recall)

    def evaluate_detections(self, detections):
        raise NotImplementedError
