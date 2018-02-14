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

'''
Created on Jun 15, 2017

@author: shujon
'''
import sys
import os
import logging
import argparse
import errno
from PIL import Image
import scipy.io
import scipy.misc
import numpy as np
from mxnet.test_utils import *

logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)

DEFAULT_DATA_DIR = "datasets/caltech101/data"
DEFAULT_OUTPUT_DATA_DIR = "{}/{}".format(DEFAULT_DATA_DIR, "images32x32")
DEFAULT_MAT_FILE = "caltech101_silhouettes_28.mat"
DEFAULT_DATASET_URL = "https://people.cs.umass.edu/~marlin/data/caltech101_silhouettes_28.mat"


def convert_mat_to_images(args):
    '''convert the caltech101 mat file to images
    Examples
    --------
    python convert_data.py --dataset /home/ubuntu/datasets/caltech101/data/caltech101_silhouettes_28.mat --save_path /home/ubuntu/datasets/caltech101/data/ --invert --height 32 --width 32
    '''
    dataset = scipy.io.loadmat("{}/{}".format(args.save_path, args.dataset))

    # image pixel data
    X = dataset['X']

    # image class labels (not used in this project)
    Y = dataset['Y']

    total_image = X.shape[0]

    h=args.height
    w=args.width

    for i in range(total_image):
        img = X[i]
        img = np.reshape(img, (28, 28))
        if args.invert:
            img = (1-img)*255
        else:
            img = img*255
        img = Image.fromarray(img, 'L')
        img = img.rotate(-90)
        img = img.resize([h, w], Image.BILINEAR)
        img.save(args.save_path + '/img' + str(i) + '.png')

def parse_args():
    '''Parses input args
    '''
    parser = argparse.ArgumentParser(description='Convert the caltech101 mat file to images')
    # general
    parser.add_argument('--dataset', help='caltech101 dataset mat file path',
                        default=DEFAULT_MAT_FILE,
                        type=str)
    parser.add_argument('--save_path', help='path to save the images',
                        default=DEFAULT_OUTPUT_DATA_DIR, type=str)
    parser.add_argument('--invert',
                        help='invert the image color i.e. default shapes are black and \
                              background is white in caltech101, invert the shapes to white',
                        action='store_true')
    parser.add_argument('--height', help='height of the final image', default=32, type=int)
    parser.add_argument('--width', help='width of the final image', default=32, type=int)

    # Note if you change the height or width you will need to change the network as well,
    # as the convolution output will be different then
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    download(DEFAULT_DATASET_URL, fname=args.dataset, dirname=args.save_path, overwrite=False)
    convert_mat_to_images(args)

if __name__ == '__main__':
    sys.exit(main())
