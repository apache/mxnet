import argparse
import os

import mxnet as mx

from tools.test_rcnn import test_rcnn
from tools.test_rcnn import parse_args

if __name__ == '__main__':
    args = parse_args()
    ctx = mx.gpu(args.gpu_id)
    test_rcnn(args.image_set, args.year, args.root_path, args.devkit_path, args.prefix, args.epoch, ctx, args.vis, args.has_rpn)
