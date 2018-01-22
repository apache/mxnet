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
from evaluate.evaluate_net import evaluate_net

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate a network')
    parser.add_argument('--rec-path', dest='rec_path', help='which record file to use',
                        default=os.path.join(os.getcwd(), 'data', 'val.rec'), type=str)
    parser.add_argument('--list-path', dest='list_path', help='which list file to use',
                        default="", type=str)
    parser.add_argument('--network', dest='network', type=str, default='vgg16_reduced',
                        help='which network to use')
    parser.add_argument('--batch-size', dest='batch_size', type=int, default=32,
                        help='evaluation batch size')
    parser.add_argument('--num-class', dest='num_class', type=int, default=20,
                        help='number of classes')
    parser.add_argument('--class-names', dest='class_names', type=str,
                        default='aeroplane, bicycle, bird, boat, bottle, bus, \
                        car, cat, chair, cow, diningtable, dog, horse, motorbike, \
                        person, pottedplant, sheep, sofa, train, tvmonitor',
                        help='string of comma separated names, or text filename')
    parser.add_argument('--epoch', dest='epoch', help='epoch of pretrained model',
                        default=0, type=int)
    parser.add_argument('--prefix', dest='prefix', help='load model prefix',
                        default=os.path.join(os.getcwd(), 'model', 'ssd_'), type=str)
    parser.add_argument('--gpus', dest='gpu_id', help='GPU devices to evaluate with',
                        default='0', type=str)
    parser.add_argument('--cpu', dest='cpu', help='use cpu to evaluate, this can be slow',
                        action='store_true')
    parser.add_argument('--data-shape', dest='data_shape', type=int, default=300,
                        help='set image shape')
    parser.add_argument('--mean-r', dest='mean_r', type=float, default=123,
                        help='red mean value')
    parser.add_argument('--mean-g', dest='mean_g', type=float, default=117,
                        help='green mean value')
    parser.add_argument('--mean-b', dest='mean_b', type=float, default=104,
                        help='blue mean value')
    parser.add_argument('--nms', dest='nms_thresh', type=float, default=0.45,
                        help='non-maximum suppression threshold')
    parser.add_argument('--overlap', dest='overlap_thresh', type=float, default=0.5,
                        help='evaluation overlap threshold')
    parser.add_argument('--force', dest='force_nms', action='store_true',
                        help='force non-maximum suppression on different class')
    parser.add_argument('--use-difficult', dest='use_difficult', action='store_true',
                        help='use difficult ground-truths in evaluation')
    parser.add_argument('--no-voc07', dest='use_voc07_metric', action='store_false',
                        help='dont use PASCAL VOC 07 metric')
    parser.add_argument('--deploy', dest='deploy_net', help='Load network from model',
                        action='store_true', default=False)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    # choose ctx
    if args.cpu:
        ctx = mx.cpu()
    else:
        ctx = [mx.gpu(int(i)) for i in args.gpu_id.split(',')]
    # parse # classes and class_names if applicable
    num_class = args.num_class
    if len(args.class_names) > 0:
        if os.path.isfile(args.class_names):
                # try to open it to read class names
                with open(args.class_names, 'r') as f:
                    class_names = [l.strip() for l in f.readlines()]
        else:
            class_names = [c.strip() for c in args.class_names.split(',')]
        assert len(class_names) == num_class
        for name in class_names:
            assert len(name) > 0
    else:
        class_names = None

    network = None if args.deploy_net else args.network
    if args.prefix.endswith('_'):
        prefix = args.prefix + args.network
    else:
        prefix = args.prefix
    evaluate_net(network, args.rec_path, num_class,
                 (args.mean_r, args.mean_g, args.mean_b), args.data_shape,
                 prefix, args.epoch, ctx, batch_size=args.batch_size,
                 path_imglist=args.list_path, nms_thresh=args.nms_thresh,
                 force_nms=args.force_nms, ovp_thresh=args.overlap_thresh,
                 use_difficult=args.use_difficult, class_names=class_names,
                 voc07_metric=args.use_voc07_metric)
