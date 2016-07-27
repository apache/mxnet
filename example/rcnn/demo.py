import argparse
import os
import numpy as np
import cv2

import mxnet as mx

from helper.processing.image_processing import resize, transform
from helper.processing.nms import nms
from rcnn.config import config
from rcnn.detector import Detector
from rcnn.symbol import get_vgg_test
from rcnn.tester import vis_all_detection
from utils.load_model import load_param


def get_net(prefix, epoch, ctx):
    args, auxs = load_param(prefix, epoch, convert=True, ctx=ctx)
    sym = get_vgg_test()
    detector = Detector(sym, ctx, args, auxs)
    return detector


CLASSES = ('__background__',
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor')


def demo_net(detector, image_name):
    """
    wrapper for detector
    :param detector: Detector
    :param image_name: image name
    :return: None
    """
    config.TEST.HAS_RPN = True
    assert os.path.exists(image_name), image_name + ' not found'
    im = cv2.imread(image_name)
    im_array, im_scale = resize(im, config.SCALES[0], config.MAX_SIZE)
    im_array = transform(im_array, config.PIXEL_MEANS)
    im_info = np.array([[im_array.shape[2], im_array.shape[3], im_scale]], dtype=np.float32)

    scores, boxes = detector.im_detect(im_array, im_info)

    all_boxes = [[] for _ in CLASSES]
    CONF_THRESH = 0.8
    NMS_THRESH = 0.3
    for cls in CLASSES:
        cls_ind = CLASSES.index(cls)
        cls_boxes = boxes[:, 4 * cls_ind:4 * (cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        keep = np.where(cls_scores >= CONF_THRESH)[0]
        cls_boxes = cls_boxes[keep, :]
        cls_scores = cls_scores[keep]
        dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets.astype(np.float32), NMS_THRESH)
        all_boxes[cls_ind] = dets[keep, :]

    boxes_this_image = [[]] + [all_boxes[j] for j in range(1, len(CLASSES))]
    vis_all_detection(im_array, boxes_this_image, CLASSES, 0)


def parse_args():
    parser = argparse.ArgumentParser(description='Demonstrate a Faster R-CNN network')
    parser.add_argument('--image', dest='image', help='custom image', type=str)
    parser.add_argument('--prefix', dest='prefix', help='saved model prefix', type=str)
    parser.add_argument('--epoch', dest='epoch', help='epoch of pretrained model', type=int)
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device to test with',
                        default=0, type=int)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    ctx = mx.gpu(args.gpu_id)
    detector = get_net(args.prefix, args.epoch, ctx)
    demo_net(detector, args.image)
    demo_net(detector, args.image)
