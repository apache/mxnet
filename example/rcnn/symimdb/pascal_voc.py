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
import numpy as np

from symnet.logger import logger
from .imdb import IMDB


class PascalVOC(IMDB):
    classes = ['__background__',  # always index 0
               'aeroplane', 'bicycle', 'bird', 'boat',
               'bottle', 'bus', 'car', 'cat', 'chair',
               'cow', 'diningtable', 'dog', 'horse',
               'motorbike', 'person', 'pottedplant',
               'sheep', 'sofa', 'train', 'tvmonitor']

    def __init__(self, image_set, root_path, devkit_path):
        """
        fill basic information to initialize imdb
        :param image_set: 2007_trainval, 2007_test, etc
        :param root_path: 'data', will write 'cache'
        :param devkit_path: 'data/VOCdevkit', load data and write results
        """
        super(PascalVOC, self).__init__('voc_' + image_set, root_path)

        year, image_set = image_set.split('_')
        self._config = {'comp_id': 'comp4',
                        'use_diff': False,
                        'min_size': 2}
        self._class_to_ind = dict(zip(self.classes, range(self.num_classes)))
        self._image_index_file = os.path.join(devkit_path, 'VOC' + year, 'ImageSets', 'Main', image_set + '.txt')
        self._image_file_tmpl = os.path.join(devkit_path, 'VOC' + year, 'JPEGImages', '{}.jpg')
        self._image_anno_tmpl = os.path.join(devkit_path, 'VOC' + year, 'Annotations', '{}.xml')

        # results
        result_folder = os.path.join(devkit_path, 'results', 'VOC' + year, 'Main')
        if not os.path.exists(result_folder):
            os.makedirs(result_folder)
        self._result_file_tmpl = os.path.join(result_folder, 'comp4_det_' + image_set + '_{}.txt')

        # get roidb
        self._roidb = self._get_cached('roidb', self._load_gt_roidb)
        logger.info('%s num_images %d' % (self.name, self.num_images))

    def _load_gt_roidb(self):
        image_index = self._load_image_index()
        gt_roidb = [self._load_annotation(index) for index in image_index]
        return gt_roidb

    def _load_image_index(self):
        with open(self._image_index_file) as f:
            image_set_index = [x.strip() for x in f.readlines()]
        return image_set_index

    def _load_annotation(self, index):
        # store original annotation as orig_objs
        height, width, orig_objs = self._parse_voc_anno(self._image_anno_tmpl.format(index))

        # filter difficult objects
        if not self._config['use_diff']:
            non_diff_objs = [obj for obj in orig_objs if obj['difficult'] == 0]
            objs = non_diff_objs
        else:
            objs = orig_objs
        num_objs = len(objs)

        boxes = np.zeros((num_objs, 4), dtype=np.uint16)
        gt_classes = np.zeros((num_objs,), dtype=np.int32)
        # Load object bounding boxes into a data frame.
        for ix, obj in enumerate(objs):
            # Make pixel indexes 0-based
            x1 = obj['bbox'][0] - 1
            y1 = obj['bbox'][1] - 1
            x2 = obj['bbox'][2] - 1
            y2 = obj['bbox'][3] - 1
            cls = self._class_to_ind[obj['name'].lower().strip()]
            boxes[ix, :] = [x1, y1, x2, y2]
            gt_classes[ix] = cls

        roi_rec = {'index': index,
                   'objs': orig_objs,
                   'image': self._image_file_tmpl.format(index),
                   'height': height,
                   'width': width,
                   'boxes': boxes,
                   'gt_classes': gt_classes,
                   'flipped': False}
        return roi_rec

    @staticmethod
    def _parse_voc_anno(filename):
        import xml.etree.ElementTree as ET
        tree = ET.parse(filename)
        height = int(tree.find('size').find('height').text)
        width = int(tree.find('size').find('width').text)
        objects = []
        for obj in tree.findall('object'):
            obj_dict = dict()
            obj_dict['name'] = obj.find('name').text
            obj_dict['difficult'] = int(obj.find('difficult').text)
            bbox = obj.find('bndbox')
            obj_dict['bbox'] = [int(float(bbox.find('xmin').text)),
                                int(float(bbox.find('ymin').text)),
                                int(float(bbox.find('xmax').text)),
                                int(float(bbox.find('ymax').text))]
            objects.append(obj_dict)
        return height, width, objects

    def _evaluate_detections(self, detections, use_07_metric=True, **kargs):
        self._write_pascal_results(detections)
        self._do_python_eval(detections, use_07_metric)

    def _write_pascal_results(self, all_boxes):
        for cls_ind, cls in enumerate(self.classes):
            if cls == '__background__':
                continue
            logger.info('Writing %s VOC results file' % cls)
            filename = self._result_file_tmpl.format(cls)
            with open(filename, 'wt') as f:
                for im_ind, roi_rec in enumerate(self.roidb):
                    index = roi_rec['index']
                    dets = all_boxes[cls_ind][im_ind]
                    if len(dets) == 0:
                        continue
                    # the VOCdevkit expects 1-based indices
                    for k in range(dets.shape[0]):
                        f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                                format(index, dets[k, -1],
                                       dets[k, 0] + 1, dets[k, 1] + 1, dets[k, 2] + 1, dets[k, 3] + 1))

    def _do_python_eval(self, all_boxes, use_07_metric):
        aps = []
        for cls_ind, cls in enumerate(self.classes):
            if cls == '__background__':
                continue
            # class_anno is a dict [image_index, [bbox, difficult, det]]
            class_anno = {}
            npos = 0
            for roi_rec in self.roidb:
                index = roi_rec['index']
                objects = [obj for obj in roi_rec['objs'] if obj['name'] == cls]
                bbox = np.array([x['bbox'] for x in objects])
                difficult = np.array([x['difficult'] for x in objects]).astype(np.bool)
                det = [False] * len(objects)  # stand for detected
                npos = npos + sum(~difficult)
                class_anno[index] = {'bbox': bbox,
                                     'difficult': difficult,
                                     'det': det}

            # bbox is 2d array of all detections, corresponding to each image_id
            image_ids = []
            bbox = []
            confidence = []
            for im_ind, dets in enumerate(all_boxes[cls_ind]):
                for k in range(dets.shape[0]):
                    image_ids.append(self.roidb[im_ind]['index'])
                    bbox.append([dets[k, 0] + 1, dets[k, 1] + 1, dets[k, 2] + 1, dets[k, 3] + 1])
                    confidence.append(dets[k, -1])
            bbox = np.array(bbox)
            confidence = np.array(confidence)

            rec, prec, ap = self.voc_eval(class_anno, npos, image_ids, bbox, confidence,
                                          ovthresh=0.5, use_07_metric=use_07_metric)
            aps.append(ap)

        for cls, ap in zip(self.classes, aps):
            logger.info('AP for {} = {:.4f}'.format(cls, ap))
        logger.info('Mean AP = {:.4f}'.format(np.mean(aps)))

    @staticmethod
    def voc_eval(class_anno, npos, image_ids, bbox, confidence, ovthresh=0.5, use_07_metric=False):
        # sort by confidence
        if bbox.shape[0] > 0:
            sorted_inds = np.argsort(-confidence)
            sorted_scores = np.sort(-confidence)
            bbox = bbox[sorted_inds, :]
            image_ids = [image_ids[x] for x in sorted_inds]

        # go down detections and mark true positives and false positives
        nd = len(image_ids)
        tp = np.zeros(nd)
        fp = np.zeros(nd)
        for d in range(nd):
            r = class_anno[image_ids[d]]
            bb = bbox[d, :].astype(float)
            ovmax = -np.inf
            bbgt = r['bbox'].astype(float)

            if bbgt.size > 0:
                # compute overlaps
                # intersection
                ixmin = np.maximum(bbgt[:, 0], bb[0])
                iymin = np.maximum(bbgt[:, 1], bb[1])
                ixmax = np.minimum(bbgt[:, 2], bb[2])
                iymax = np.minimum(bbgt[:, 3], bb[3])
                iw = np.maximum(ixmax - ixmin + 1., 0.)
                ih = np.maximum(iymax - iymin + 1., 0.)
                inters = iw * ih

                # union
                uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
                       (bbgt[:, 2] - bbgt[:, 0] + 1.) *
                       (bbgt[:, 3] - bbgt[:, 1] + 1.) - inters)

                overlaps = inters / uni
                ovmax = np.max(overlaps)
                jmax = np.argmax(overlaps)

            if ovmax > ovthresh:
                if not r['difficult'][jmax]:
                    if not r['det'][jmax]:
                        tp[d] = 1.
                        r['det'][jmax] = 1
                    else:
                        fp[d] = 1.
            else:
                fp[d] = 1.

        # compute precision recall
        fp = np.cumsum(fp)
        tp = np.cumsum(tp)
        rec = tp / float(npos)
        # avoid division by zero in case first detection matches a difficult ground ruth
        prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
        ap = PascalVOC.voc_ap(rec, prec, use_07_metric)

        return rec, prec, ap

    @staticmethod
    def voc_ap(rec, prec, use_07_metric=False):
        if use_07_metric:
            ap = 0.
            for t in np.arange(0., 1.1, 0.1):
                if np.sum(rec >= t) == 0:
                    p = 0
                else:
                    p = np.max(prec[rec >= t])
                ap += p / 11.
        else:
            # append sentinel values at both ends
            mrec = np.concatenate(([0.], rec, [1.]))
            mpre = np.concatenate(([0.], prec, [0.]))

            # compute precision integration ladder
            for i in range(mpre.size - 1, 0, -1):
                mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

            # look for recall value changes
            i = np.where(mrec[1:] != mrec[:-1])[0]

            # sum (\delta recall) * prec
            ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
        return ap
