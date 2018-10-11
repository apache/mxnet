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

import mxnet as mx
import random
from mxnet.io import DataBatch, DataIter
import numpy as np

def add_data_args(parser):
    data = parser.add_argument_group('Data', 'the input images')
    data.add_argument('--data-train', type=str, help='the training data')
    data.add_argument('--data-train-idx', type=str, default='', help='the index of training data')
    data.add_argument('--data-val', type=str, help='the validation data')
    data.add_argument('--data-val-idx', type=str, default='', help='the index of validation data')
    data.add_argument('--rgb-mean', type=str, default='123.68,116.779,103.939',
                      help='a tuple of size 3 for the mean rgb')
    data.add_argument('--rgb-std', type=str, default='1,1,1',
                      help='a tuple of size 3 for the std rgb')
    data.add_argument('--pad-size', type=int, default=0,
                      help='padding the input image')
    data.add_argument('--fill-value', type=int, default=127,
                      help='Set the padding pixels value to fill_value')
    data.add_argument('--image-shape', type=str,
                      help='the image shape feed into the network, e.g. (3,224,224)')
    data.add_argument('--num-classes', type=int, help='the number of classes')
    data.add_argument('--num-examples', type=int, help='the number of training examples')
    data.add_argument('--data-nthreads', type=int, default=4,
                      help='number of threads for data decoding')
    data.add_argument('--benchmark', type=int, default=0,
                      help='if 1, then feed the network with synthetic data')
    return data

def add_data_aug_args(parser):
    aug = parser.add_argument_group(
        'Image augmentations', 'implemented in src/io/image_aug_default.cc')
    aug.add_argument('--random-crop', type=int, default=0,
                     help='if or not randomly crop the image')
    aug.add_argument('--random-mirror', type=int, default=0,
                     help='if or not randomly flip horizontally')
    aug.add_argument('--max-random-h', type=int, default=0,
                     help='max change of hue, whose range is [0, 180]')
    aug.add_argument('--max-random-s', type=int, default=0,
                     help='max change of saturation, whose range is [0, 255]')
    aug.add_argument('--max-random-l', type=int, default=0,
                     help='max change of intensity, whose range is [0, 255]')
    aug.add_argument('--min-random-aspect-ratio', type=float, default=None,
                     help='min value of aspect ratio, whose value is either None or a positive value.')
    aug.add_argument('--max-random-aspect-ratio', type=float, default=0,
                     help='max value of aspect ratio. If min_random_aspect_ratio is None, '
                          'the aspect ratio range is [1-max_random_aspect_ratio, '
                          '1+max_random_aspect_ratio], otherwise it is '
                          '[min_random_aspect_ratio, max_random_aspect_ratio].')
    aug.add_argument('--max-random-rotate-angle', type=int, default=0,
                     help='max angle to rotate, whose range is [0, 360]')
    aug.add_argument('--max-random-shear-ratio', type=float, default=0,
                     help='max ratio to shear, whose range is [0, 1]')
    aug.add_argument('--max-random-scale', type=float, default=1,
                     help='max ratio to scale')
    aug.add_argument('--min-random-scale', type=float, default=1,
                     help='min ratio to scale, should >= img_size/input_shape. '
                          'otherwise use --pad-size')
    aug.add_argument('--max-random-area', type=float, default=1,
                     help='max area to crop in random resized crop, whose range is [0, 1]')
    aug.add_argument('--min-random-area', type=float, default=1,
                     help='min area to crop in random resized crop, whose range is [0, 1]')
    aug.add_argument('--min-crop-size', type=int, default=-1,
                     help='Crop both width and height into a random size in '
                          '[min_crop_size, max_crop_size]')
    aug.add_argument('--max-crop-size', type=int, default=-1,
                     help='Crop both width and height into a random size in '
                          '[min_crop_size, max_crop_size]')
    aug.add_argument('--brightness', type=float, default=0,
                     help='brightness jittering, whose range is [0, 1]')
    aug.add_argument('--contrast', type=float, default=0,
                     help='contrast jittering, whose range is [0, 1]')
    aug.add_argument('--saturation', type=float, default=0,
                     help='saturation jittering, whose range is [0, 1]')
    aug.add_argument('--pca-noise', type=float, default=0,
                     help='pca noise, whose range is [0, 1]')
    aug.add_argument('--random-resized-crop', type=int, default=0,
                     help='whether to use random resized crop')
    return aug

class SyntheticDataIter(DataIter):
    def __init__(self, num_classes, data_shape, max_iter, dtype):
        self.batch_size = data_shape[0]
        self.cur_iter = 0
        self.max_iter = max_iter
        self.dtype = dtype
        label = np.random.randint(0, num_classes, [self.batch_size,])
        data = np.random.uniform(-1, 1, data_shape)
        self.data = mx.nd.array(data, dtype=self.dtype, ctx=mx.Context('cpu_pinned', 0))
        self.label = mx.nd.array(label, dtype=self.dtype, ctx=mx.Context('cpu_pinned', 0))
    def __iter__(self):
        return self
    @property
    def provide_data(self):
        return [mx.io.DataDesc('data', self.data.shape, self.dtype)]
    @property
    def provide_label(self):
        return [mx.io.DataDesc('softmax_label', (self.batch_size,), self.dtype)]
    def next(self):
        self.cur_iter += 1
        if self.cur_iter <= self.max_iter:
            return DataBatch(data=(self.data,),
                             label=(self.label,),
                             pad=0,
                             index=None,
                             provide_data=self.provide_data,
                             provide_label=self.provide_label)
        else:
            raise StopIteration
    def __next__(self):
        return self.next()
    def reset(self):
        self.cur_iter = 0

def get_rec_iter(args, kv=None):
    image_shape = tuple([int(l) for l in args.image_shape.split(',')])
    if 'benchmark' in args and args.benchmark:
        data_shape = (args.batch_size,) + image_shape
        train = SyntheticDataIter(args.num_classes, data_shape,
                args.num_examples / args.batch_size, np.float32)
        return (train, None)
    if kv:
        (rank, nworker) = (kv.rank, kv.num_workers)
    else:
        (rank, nworker) = (0, 1)
    rgb_mean = [float(i) for i in args.rgb_mean.split(',')]
    rgb_std = [float(i) for i in args.rgb_std.split(',')]
    train = mx.io.ImageRecordIter(
        path_imgrec         = args.data_train,
        path_imgidx         = args.data_train_idx,
        label_width         = 1,
        mean_r              = rgb_mean[0],
        mean_g              = rgb_mean[1],
        mean_b              = rgb_mean[2],
        std_r               = rgb_std[0],
        std_g               = rgb_std[1],
        std_b               = rgb_std[2],
        data_name           = 'data',
        label_name          = 'softmax_label',
        data_shape          = image_shape,
        batch_size          = args.batch_size,
        rand_crop           = args.random_crop,
        max_random_scale    = args.max_random_scale,
        pad                 = args.pad_size,
        fill_value          = args.fill_value,
        random_resized_crop = args.random_resized_crop,
        min_random_scale    = args.min_random_scale,
        max_aspect_ratio    = args.max_random_aspect_ratio,
        min_aspect_ratio    = args.min_random_aspect_ratio,
        max_random_area     = args.max_random_area,
        min_random_area     = args.min_random_area,
        min_crop_size       = args.min_crop_size,
        max_crop_size       = args.max_crop_size,
        brightness          = args.brightness,
        contrast            = args.contrast,
        saturation          = args.saturation,
        pca_noise           = args.pca_noise,
        random_h            = args.max_random_h,
        random_s            = args.max_random_s,
        random_l            = args.max_random_l,
        max_rotate_angle    = args.max_random_rotate_angle,
        max_shear_ratio     = args.max_random_shear_ratio,
        rand_mirror         = args.random_mirror,
        preprocess_threads  = args.data_nthreads,
        shuffle             = True,
        num_parts           = nworker,
        part_index          = rank)
    if args.data_val is None:
        return (train, None)
    val = mx.io.ImageRecordIter(
        path_imgrec         = args.data_val,
        path_imgidx         = args.data_val_idx,
        label_width         = 1,
        mean_r              = rgb_mean[0],
        mean_g              = rgb_mean[1],
        mean_b              = rgb_mean[2],
        std_r               = rgb_std[0],
        std_g               = rgb_std[1],
        std_b               = rgb_std[2],
        data_name           = 'data',
        label_name          = 'softmax_label',
        batch_size          = args.batch_size,
        data_shape          = image_shape,
        preprocess_threads  = args.data_nthreads,
        rand_crop           = False,
        rand_mirror         = False,
        num_parts           = nworker,
        part_index          = rank)
    return (train, val)
