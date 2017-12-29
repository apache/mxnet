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

"""
given a pascal voc imdb, compute mAP
"""

from ..logger import logger
import numpy as np
import os
try:
    import cPickle as pickle
except ImportError:
    import pickle

def parse_voc_rec(filename):
    """
    parse pascal voc record into a dictionary
    :param filename: xml file path
    :return: list of dict
    """
    import xml.etree.ElementTree as ET
    tree = ET.parse(filename)
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
    return objects


def voc_ap(rec, prec, use_07_metric=False):
    """
    average precision calculations
    [precision integrated to recall]
    :param rec: recall
    :param prec: precision
    :param use_07_metric: 2007 metric is 11-recall-point based AP
    :return: average precision
    """
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


def voc_eval(detpath, annopath, imageset_file, classname, annocache, ovthresh=0.5, use_07_metric=False):
    """
    pascal voc evaluation
    :param detpath: detection results detpath.format(classname)
    :param annopath: annotations annopath.format(classname)
    :param imageset_file: text file containing list of images
    :param classname: category name
    :param annocache: caching annotations
    :param ovthresh: overlap threshold
    :param use_07_metric: whether to use voc07's 11 point ap computation
    :return: rec, prec, ap
    """
    with open(imageset_file, 'r') as f:
        lines = f.readlines()
    image_filenames = [x.strip() for x in lines]

    # load annotations from cache
    if not os.path.isfile(annocache):
        recs = {}
        for ind, image_filename in enumerate(image_filenames):
            recs[image_filename] = parse_voc_rec(annopath.format(image_filename))
            if ind % 100 == 0:
                logger.info('reading annotations for %d/%d' % (ind + 1, len(image_filenames)))
        logger.info('saving annotations cache to %s' % annocache)
        with open(annocache, 'wb') as f:
            pickle.dump(recs, f, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with open(annocache, 'rb') as f:
            recs = pickle.load(f)

    # extract objects in :param classname:
    class_recs = {}
    npos = 0
    for image_filename in image_filenames:
        objects = [obj for obj in recs[image_filename] if obj['name'] == classname]
        bbox = np.array([x['bbox'] for x in objects])
        difficult = np.array([x['difficult'] for x in objects]).astype(np.bool)
        det = [False] * len(objects)  # stand for detected
        npos = npos + sum(~difficult)
        class_recs[image_filename] = {'bbox': bbox,
                                      'difficult': difficult,
                                      'det': det}

    # read detections
    detfile = detpath.format(classname)
    with open(detfile, 'r') as f:
        lines = f.readlines()

    splitlines = [x.strip().split(' ') for x in lines]
    image_ids = [x[0] for x in splitlines]
    confidence = np.array([float(x[1]) for x in splitlines])
    bbox = np.array([[float(z) for z in x[2:]] for x in splitlines])

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
        r = class_recs[image_ids[d]]
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
    ap = voc_ap(rec, prec, use_07_metric)

    return rec, prec, ap
