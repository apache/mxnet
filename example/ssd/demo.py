#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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

import argparse
import tools.find_mxnet
import mxnet as mx
import os
import sys
from detect.detector import Detector
from symbol.symbol_factory import get_symbol
from dataset.cv2Iterator import CameraIterator
import logging
import cv2

def get_detector(net, prefix, epoch, data_shape, mean_pixels, ctx, num_class,
                 nms_thresh=0.5, force_nms=True, nms_topk=400):
    """
    wrapper for initialize a detector

    Parameters:
    ----------
    net : str
        test network name
    prefix : str
        load model prefix
    epoch : int
        load model epoch
    data_shape : int
        resize image shape
    mean_pixels : tuple (float, float, float)
        mean pixel values (R, G, B)
    ctx : mx.ctx
        running context, mx.cpu() or mx.gpu(?)
    num_class : int
        number of classes
    nms_thresh : float
        non-maximum suppression threshold
    force_nms : bool
        force suppress different categories
    """
    if net is not None:
        if isinstance(data_shape, tuple):
            data_shape = data_shape[0]
        net = get_symbol(net, data_shape, num_classes=num_class, nms_thresh=nms_thresh,
            force_nms=force_nms, nms_topk=nms_topk)
    detector = Detector(net, prefix, epoch, data_shape, mean_pixels, ctx=ctx)
    return detector

def parse_args():
    parser = argparse.ArgumentParser(description='Single-shot detection network demo')
    parser.add_argument('--network', dest='network', type=str, default='resnet50',
                        help='which network to use')
    parser.add_argument('--images', dest='images', type=str, default='./data/demo/dog.jpg',
                        help='run demo with images, use comma to seperate multiple images')
    parser.add_argument('--dir', dest='dir', nargs='?',
                        help='demo image directory, optional', type=str)
    parser.add_argument('--ext', dest='extension', help='image extension, optional',
                        type=str, nargs='?')
    parser.add_argument('--epoch', dest='epoch', help='epoch of trained model',
                        default=0, type=int)
    parser.add_argument('--batch-size', dest='batch_size', help='batch size',
                        default=1, type=int)
    parser.add_argument('--prefix', dest='prefix', help='trained model prefix',
                        default=os.path.join(os.getcwd(), 'model', 'ssd_'),
                        type=str)
    parser.add_argument('--cpu', dest='cpu', help='(override GPU) use CPU to detect',
                        action='store_true', default=False)
    parser.add_argument('--gpu', dest='gpu_id', type=int, default=0,
                        help='GPU device id to detect with')
    parser.add_argument('--data-shape', dest='data_shape', type=str, default='512',
                        help='set image shape')
    parser.add_argument('--mean-r', dest='mean_r', type=float, default=123,
                        help='red mean value')
    parser.add_argument('--mean-g', dest='mean_g', type=float, default=117,
                        help='green mean value')
    parser.add_argument('--mean-b', dest='mean_b', type=float, default=104,
                        help='blue mean value')
    parser.add_argument('--thresh', dest='thresh', type=float, default=0.5,
                        help='object visualize score threshold, default 0.6')
    parser.add_argument('--nms', dest='nms_thresh', type=float, default=0.5,
                        help='non-maximum suppression threshold, default 0.5')
    parser.add_argument('--no-force', dest='force_nms', action='store_false',
                        help='dont force non-maximum suppression on different class')
    parser.add_argument('--no-timer', dest='show_timer', action='store_false',
                        help='dont show detection time')
    parser.add_argument('--deploy', dest='deploy_net', action='store_true', default=False,
                        help='Load network from json file, rather than from symbol')
    parser.add_argument('--class-names', dest='class_names', type=str,
                        default='aeroplane, bicycle, bird, boat, bottle, bus, \
                        car, cat, chair, cow, diningtable, dog, horse, motorbike, \
                        person, pottedplant, sheep, sofa, train, tvmonitor',
                        help='string of comma separated names, or text filename')
    parser.add_argument('--camera', action='store_true',
                        help="use camera for image capturing")
    parser.add_argument('--frame-resize', type=str, default=None,
                        help="resize camera frame to x,y pixels or a float scaling factor")
    args = parser.parse_args()
    return args

def parse_class_names(class_names):
    """ parse # classes and class_names if applicable """
    if len(class_names) > 0:
        if os.path.isfile(class_names):
            # try to open it to read class names
            with open(class_names, 'r') as f:
                class_names = [l.strip() for l in f.readlines()]
        else:
            class_names = [c.strip() for c in class_names.split(',')]
        for name in class_names:
            assert len(name) > 0
    else:
        raise RuntimeError("No valid class_name provided...")
    return class_names

def parse_frame_resize(x):
    if not x:
        return x
    x = list(map(float, x.strip().split(',')))
    assert len(x) >= 1 and len(x) <= 2, "frame_resize should be a float scaling factor or a tuple of w,h pixels"
    if len(x) == 1:
        x = x[0]
    return x

def parse_data_shape(data_shape_str):
    """Parse string to tuple or int"""
    ds = data_shape_str.strip().split(',')
    if len(ds) == 1:
        data_shape = (int(ds[0]), int(ds[0]))
    elif len(ds) == 2:
        data_shape = (int(ds[0]), int(ds[1]))
    else:
        raise ValueError("Unexpected data_shape: %s", data_shape_str)
    return data_shape

def draw_detection(frame, det, class_names):
    (klass, score, x0, y0, x1, y1) = det
    klass_name = class_names[int(klass)]
    h = frame.shape[0]
    w = frame.shape[1]
    # denormalize detections from [0,1] to the frame size
    p0 = tuple(map(int, (x0*w,y0*h)))
    p1 = tuple(map(int, (x1*w,y1*h)))
    logging.info("detection: %s %s", klass_name, score)
    cv2.rectangle(frame, p0, p1, (0,0,255), 2)
    # Where to draw the text, a few pixels above the top y coordinate
    tp0 = (p0[0], p0[1]-5)
    draw_text = "{} {}".format(klass_name, score)
    cv2.putText(frame, draw_text, tp0, cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, (0,0,255))


def network_path(prefix, network, data_shape):
    return "{}{}_{}".format(prefix, network, data_shape)

def run_camera(args,ctx):
    assert args.batch_size == 1, "only batch size of 1 is supported"
    logging.info("Detection threshold is {}".format(args.thresh))
    iter = CameraIterator(frame_resize=parse_frame_resize(args.frame_resize))
    class_names = parse_class_names(args.class_names)
    mean_pixels = (args.mean_r, args.mean_g, args.mean_b)
    data_shape = int(args.data_shape)
    batch_size = int(args.batch_size)
    detector = Detector(
        get_symbol(args.network, data_shape, num_classes=len(class_names)),
        network_path(args.prefix, args.network, data_shape),
        args.epoch,
        data_shape,
        mean_pixels,
        batch_size,
        ctx
    )
    for frame in iter:
        logging.info("Frame info: shape %s type %s", frame.shape, frame.dtype)
        logging.info("Generating batch")
        data_batch = detector.create_batch(frame)
        logging.info("Detecting objects")
        detections_batch = detector.detect_batch(data_batch)
        #detections = [mx.nd.array((1,1,0.2,0.2,0.4,0.4))]
        detections = detections_batch[0]
        logging.info("%d detections", len(detections))
        for det in detections:
            obj = det.asnumpy()
            (klass, score, x0, y0, x1, y1) = obj
            if score > args.thresh:
                draw_detection(frame, obj, class_names)
        cv2.imshow('frame', frame)

def run_images(args,ctx):
    # parse image list
    image_list = [i.strip() for i in args.images.split(',')]
    assert len(image_list) > 0, "No valid image specified to detect"

    network = None if args.deploy_net else args.network
    class_names = parse_class_names(args.class_names)
    data_shape = parse_data_shape(args.data_shape)
    if args.prefix.endswith('_'):
        prefix = args.prefix + args.network + '_' + str(data_shape[0])
    else:
        prefix = args.prefix
    detector = get_detector(network, prefix, args.epoch,
                            data_shape,
                            (args.mean_r, args.mean_g, args.mean_b),
                            ctx, len(class_names), args.nms_thresh, args.force_nms)
    # run detection
    detector.detect_and_visualize(image_list, args.dir, args.extension,
                                  class_names, args.thresh, args.show_timer)

def main():
    logging.getLogger().setLevel(logging.INFO)
    logging.basicConfig(format='%(asctime)-15s %(message)s')
    args = parse_args()
    if args.cpu:
        ctx = mx.cpu()
    else:
        ctx = mx.gpu(args.gpu_id)

    if args.camera:
        run_camera(args, ctx)
    else:
        run_images(args, ctx)
    return 0

if __name__ == '__main__':
    sys.exit(main())

