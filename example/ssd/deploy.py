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

from __future__ import print_function
import argparse
import tools.find_mxnet
import mxnet as mx
import os
import importlib
import sys
from symbol.symbol_factory import get_symbol

def parse_args():
    parser = argparse.ArgumentParser(description='Convert a trained model to deploy model')
    parser.add_argument('--network', dest='network', type=str, default='vgg16_reduced',
                        help='which network to use')
    parser.add_argument('--epoch', dest='epoch', help='epoch of trained model',
                        default=0, type=int)
    parser.add_argument('--prefix', dest='prefix', help='trained model prefix',
                        default=os.path.join(os.getcwd(), 'model', 'ssd_'), type=str)
    parser.add_argument('--data-shape', dest='data_shape', type=int, default=300,
                        help='data shape')
    parser.add_argument('--num-class', dest='num_classes', help='number of classes',
                        default=20, type=int)
    parser.add_argument('--nms', dest='nms_thresh', type=float, default=0.5,
                        help='non-maximum suppression threshold, default 0.5')
    parser.add_argument('--no-force', dest='force_nms', action='store_false',
                        help='dont force non-maximum suppression on different class')
    parser.add_argument('--topk', dest='nms_topk', type=int, default=400,
                        help='apply nms only to top k detections based on scores.')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    net = get_symbol(args.network, args.data_shape,
        num_classes=args.num_classes, nms_thresh=args.nms_thresh,
        force_suppress=args.force_nms, nms_topk=args.nms_topk)
    if args.prefix.endswith('_'):
        prefix = args.prefix + args.network + '_' + str(args.data_shape)
    else:
        prefix = args.prefix
    _, arg_params, aux_params = mx.model.load_checkpoint(prefix, args.epoch)
    # new name
    tmp = prefix.rsplit('/', 1)
    save_prefix = '/deploy_'.join(tmp)
    mx.model.save_checkpoint(save_prefix, args.epoch, net, arg_params, aux_params)
    print("Saved model: {}-{:04d}.params".format(save_prefix, args.epoch))
    print("Saved symbol: {}-symbol.json".format(save_prefix))
