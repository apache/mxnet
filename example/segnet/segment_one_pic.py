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
import mxnet as mx
import time
import os
import logging
import numpy as np
from PIL import Image
from data_iter import FileIter
from common import contrib_metrics

def getpallete(num_cls):
    """
    this function is to get the colormap for visualizing the segmentation mask
    """
    pallete = [0] * (num_cls * 3)
    for j in xrange(0, num_cls):
        lab = j
        pallete[j*3+0] = 0
        pallete[j*3+1] = 0
        pallete[j*3+2] = 0
        i = 0
        while lab > 0:
            pallete[j*3+0] |= (((lab >> 0) & 1) << (7-i))
            pallete[j*3+1] |= (((lab >> 1) & 1) << (7-i))
            pallete[j*3+2] |= (((lab >> 2) & 1) << (7-i))
            i = i + 1
            lab >>= 3
    return pallete

def read_img(args):
    """
    read picture to numpy.array
    """
    img = mx.image.imread(os.path.join(args.image_path, args.image_name))
    img = img.astype('float32')
    mean = mx.nd.array([float(i) for i in args.rgb_mean.split(',')])
    reshaped_mean = mean.reshape((1, 1, 3))
    img = img - reshaped_mean
    img = mx.nd.moveaxis(img, 2, 0)
    img = mx.nd.expand_dims(img, axis=0)
    return img

def read_label(args):
    """
    read label to numpy.array
    """
    label = mx.image.imread(os.path.join(args.label_path, args.label_name), flag=0)
    label = label.astype('float32')
    label = mx.nd.moveaxis(label, 2, 0)
    return label

def score(args):
    """
    test one pic and save the result
    """
    pallete = getpallete(256)
    # create module
    if args.gpus == '':
        devs = mx.cpu()
    else:
        devs = [mx.gpu(int(i)) for i in args.gpus.split(',')]
    # create data iterator
    img = read_img(args)
    label = read_label(args)
    data_shape = tuple([int(i) for i in args.image_shape.split(',')])
    batch_data = mx.ndarray.array(img[0])
    batch_data = batch_data.reshape((tuple([1]) + data_shape))
    data = mx.io.DataBatch([batch_data], [])
    try:
        sym, arg_params, aux_params = mx.model.load_checkpoint(args.model_prefix, args.load_epoch)
    except ():
        logging.info('model load error!')


    mod = mx.mod.Module(symbol=sym, context=devs, label_names=['softmax_label',])
    mod.bind(for_training=False,
             data_shapes=[('data', (tuple([1]) + data_shape))],
             label_shapes=[('softmax_label', (tuple([1] + [data_shape[1]*data_shape[2]])))])
    mod.set_params(arg_params, aux_params)
    metric = contrib_metrics.Accuracy(ignore_label=11)
    tic = time.time()
    mod.forward(data, is_train=False)
    out_img = np.uint8(np.squeeze(mod.get_outputs()[0].asnumpy().argmax(axis=1)))
    metric.reset()
    metric.update([label], [mod.get_outputs()[0]])
    logging.info(metric.get())
    out_img = Image.fromarray(np.uint8(label[0].asnumpy()))
    out_img.putpalette(pallete)
    out_img.save('res_pic/' + args.image_name)
    out_label = Image.fromarray(label)
    out_label.putpalette(pallete)
    out_label.save('res_pic/label' + args.image_name)
    return (1 / (time.time() - tic), )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='score a model on a dataset')
    parser.add_argument('--gpus', type=str, default='0')
    parser.add_argument('--rgb_mean', type=str, default='123.68, 116.779, 103.939')
    parser.add_argument('--image_shape', type=str, default='3,360,480')
    parser.add_argument('--data_nthreads', type=int, default=4,
                        help='number of threads for data decoding')
    parser.add_argument('--image_path', type=str, default="/data/CamVid/Image/",
                        help='data path')
    parser.add_argument('--image_name', type=str, default="0016E5_04530.png",
                        help='test data list file name in data path')
    parser.add_argument('--label_path', type=str, default="/data/CamVid/Label/",
                        help='data path')
    parser.add_argument('--label_name', type=str, default="0016E5_04530.png",
                        help='test data list file name in data path')
    parser.add_argument('--model_prefix', type=str, default='models/segnet',
                        help='model prefix')
    parser.add_argument('--load_epoch', type=int, default=70,
                        help='load the model on an epoch using the model-load-prefix')
    parser.add_argument('--log_file', type=str, default="log.txt",
                        help='the name of log file')
    args = parser.parse_args()

    (speed,) = score(args)
    logging.info('Finished with %f images per second', speed)
