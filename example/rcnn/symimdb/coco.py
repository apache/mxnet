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

import os
import json
import numpy as np
from builtins import range

from symnet.logger import logger
from .imdb import IMDB

# coco api
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


class coco(IMDB):
    classes = ['__background__',  # always index 0
               'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
               'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
               'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
               'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
               'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
               'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
               'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
               'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
               'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
               'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
               'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
               'scissors', 'teddy bear', 'hair drier', 'toothbrush']

    def __init__(self, image_set, root_path, data_path):
        """
        fill basic information to initialize imdb
        :param image_set: train2017, val2017
        :param root_path: 'data', will write 'cache'
        :param data_path: 'data/coco', load data and write results
        """
        super(coco, self).__init__('coco_' + image_set, root_path)
        # example: annotations/instances_train2017.json
        self._anno_file = os.path.join(data_path, 'annotations', 'instances_' + image_set + '.json')
        # example train2017/000000119993.jpg
        self._image_file_tmpl = os.path.join(data_path, image_set, '{}')
        # example detections_val2017_results.json
        self._result_file = os.path.join(data_path, 'detections_{}_results.json'.format(image_set))
        # get roidb
        self._roidb = self._get_cached('roidb', self._load_gt_roidb)
        logger.info('%s num_images %d' % (self.name, self.num_images))

    def _load_gt_roidb(self):
        _coco = COCO(self._anno_file)
        # deal with class names
        cats = [cat['name'] for cat in _coco.loadCats(_coco.getCatIds())]
        class_to_coco_ind = dict(zip(cats, _coco.getCatIds()))
        class_to_ind = dict(zip(self.classes, range(self.num_classes)))
        coco_ind_to_class_ind = dict([(class_to_coco_ind[cls], class_to_ind[cls])
                                     for cls in self.classes[1:]])

        image_ids = _coco.getImgIds()
        gt_roidb = [self._load_annotation(_coco, coco_ind_to_class_ind, index) for index in image_ids]
        return gt_roidb

    def _load_annotation(self, _coco, coco_ind_to_class_ind, index):
        """
        coco ann: [u'segmentation', u'area', u'iscrowd', u'image_id', u'bbox', u'category_id', u'id']
        iscrowd:
            crowd instances are handled by marking their overlaps with all categories to -1
            and later excluded in training
        bbox:
            [x1, y1, w, h]
        :param index: coco image id
        :return: roidb entry
        """
        im_ann = _coco.loadImgs(index)[0]
        filename = self._image_file_tmpl.format(im_ann['file_name'])
        width = im_ann['width']
        height = im_ann['height']

        annIds = _coco.getAnnIds(imgIds=index, iscrowd=None)
        objs = _coco.loadAnns(annIds)

        # sanitize bboxes
        valid_objs = []
        for obj in objs:
            x, y, w, h = obj['bbox']
            x1 = np.max((0, x))
            y1 = np.max((0, y))
            x2 = np.min((width - 1, x1 + np.max((0, w - 1))))
            y2 = np.min((height - 1, y1 + np.max((0, h - 1))))
            if obj['area'] > 0 and x2 >= x1 and y2 >= y1:
                obj['clean_bbox'] = [x1, y1, x2, y2]
                valid_objs.append(obj)
        objs = valid_objs
        num_objs = len(objs)

        boxes = np.zeros((num_objs, 4), dtype=np.uint16)
        gt_classes = np.zeros((num_objs,), dtype=np.int32)
        for ix, obj in enumerate(objs):
            cls = coco_ind_to_class_ind[obj['category_id']]
            boxes[ix, :] = obj['clean_bbox']
            gt_classes[ix] = cls

        roi_rec = {'index': index,
                   'image': filename,
                   'height': height,
                   'width': width,
                   'boxes': boxes,
                   'gt_classes': gt_classes,
                   'flipped': False}
        return roi_rec

    def _evaluate_detections(self, detections, **kargs):
        _coco = COCO(self._anno_file)
        self._write_coco_results(_coco, detections)
        self._do_python_eval(_coco)

    def _write_coco_results(self, _coco, detections):
        """ example results
        [{"image_id": 42,
          "category_id": 18,
          "bbox": [258.15,41.29,348.26,243.78],
          "score": 0.236}, ...]
        """
        cats = [cat['name'] for cat in _coco.loadCats(_coco.getCatIds())]
        class_to_coco_ind = dict(zip(cats, _coco.getCatIds()))
        results = []
        for cls_ind, cls in enumerate(self.classes):
            if cls == '__background__':
                continue
            logger.info('collecting %s results (%d/%d)' % (cls, cls_ind, self.num_classes - 1))
            coco_cat_id = class_to_coco_ind[cls]
            results.extend(self._coco_results_one_category(detections[cls_ind], coco_cat_id))
        logger.info('writing results json to %s' % self._result_file)
        with open(self._result_file, 'w') as f:
            json.dump(results, f, sort_keys=True, indent=4)

    def _coco_results_one_category(self, boxes, cat_id):
        results = []
        for im_ind, roi_rec in enumerate(self.roidb):
            index = roi_rec['index']
            dets = boxes[im_ind].astype(np.float)
            if len(dets) == 0:
                continue
            scores = dets[:, -1]
            xs = dets[:, 0]
            ys = dets[:, 1]
            ws = dets[:, 2] - xs + 1
            hs = dets[:, 3] - ys + 1
            result = [{'image_id': index,
                       'category_id': cat_id,
                       'bbox': [xs[k], ys[k], ws[k], hs[k]],
                       'score': scores[k]} for k in range(dets.shape[0])]
            results.extend(result)
        return results

    def _do_python_eval(self, _coco):
        coco_dt = _coco.loadRes(self._result_file)
        coco_eval = COCOeval(_coco, coco_dt)
        coco_eval.params.useSegm = False
        coco_eval.evaluate()
        coco_eval.accumulate()
        self._print_detection_metrics(coco_eval)

    def _print_detection_metrics(self, coco_eval):
        IoU_lo_thresh = 0.5
        IoU_hi_thresh = 0.95

        def _get_thr_ind(coco_eval, thr):
            ind = np.where((coco_eval.params.iouThrs > thr - 1e-5) &
                           (coco_eval.params.iouThrs < thr + 1e-5))[0][0]
            iou_thr = coco_eval.params.iouThrs[ind]
            assert np.isclose(iou_thr, thr)
            return ind

        ind_lo = _get_thr_ind(coco_eval, IoU_lo_thresh)
        ind_hi = _get_thr_ind(coco_eval, IoU_hi_thresh)

        # precision has dims (iou, recall, cls, area range, max dets)
        # area range index 0: all area ranges
        # max dets index 2: 100 per image
        precision = \
            coco_eval.eval['precision'][ind_lo:(ind_hi + 1), :, :, 0, 2]
        ap_default = np.mean(precision[precision > -1])
        logger.info('~~~~ Mean and per-category AP @ IoU=%.2f,%.2f] ~~~~' % (IoU_lo_thresh, IoU_hi_thresh))
        logger.info('%-15s %5.1f' % ('all', 100 * ap_default))
        for cls_ind, cls in enumerate(self.classes):
            if cls == '__background__':
                continue
            # minus 1 because of __background__
            precision = coco_eval.eval['precision'][ind_lo:(ind_hi + 1), :, cls_ind - 1, 0, 2]
            ap = np.mean(precision[precision > -1])
            logger.info('%-15s %5.1f' % (cls, 100 * ap))

        logger.info('~~~~ Summary metrics ~~~~')
        coco_eval.summarize()
