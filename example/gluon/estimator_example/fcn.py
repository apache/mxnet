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

# This example is compiled from
# https://github.com/d2l-ai/d2l-en/blob/master/chapter_computer-vision/fcn.md

import tarfile
import os
import sys
import argparse
import numpy as np
import mxnet as mx
from mxnet import gluon, image, init, nd
from mxnet.gluon import data as gdata, model_zoo, nn
from mxnet.gluon.estimator import estimator, event_handler

def parse_args():
    '''
    Command Line Interface
    '''
    parser = argparse.ArgumentParser(description='Train ResNet18 on Fashion-MNIST')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='training batch size per device (CPU/GPU).')
    parser.add_argument('--num-epochs', type=int, default=1,
                        help='number of training epochs.')
    parser.add_argument('--input-size', type=tuple, default=(320, 480),
                        help='size of the input image size. default is (320, 480)')
    parser.add_argument('--lr', type=float, default=0.1,
                        help='learning rate. default is 0.1')
    parser.add_argument('-j', '--num-workers', default=None, type=int,
                        help='number of preprocessing workers')
    opt = parser.parse_args()
    return opt


def FCN(num_classes=21, ctx=None):
    '''
    FCN model for semantic segmentation
    '''
    pretrained_net = model_zoo.vision.resnet18_v2(pretrained=True, ctx=ctx)

    net = nn.HybridSequential()
    for layer in pretrained_net.features[:-2]:
        net.add(layer)

    net.add(nn.Conv2D(num_classes, kernel_size=1),
            nn.Conv2DTranspose(num_classes, kernel_size=64, padding=16, strides=32))
    return net

def bilinear_kernel(in_channels, out_channels, kernel_size):
    '''
    Bilinear interpolation using transposed convolution
    '''
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:kernel_size, :kernel_size]
    filt = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)
    weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size), dtype='float32')
    weight[range(in_channels), range(out_channels), :, :] = filt
    return nd.array(weight)

def download_voc_pascal(data_dir='../data'):
    """Download the Pascal VOC2012 Dataset."""
    voc_dir = os.path.join(data_dir, 'VOCdevkit/VOC2012')
    url = ('http://host.robots.ox.ac.uk/pascal/VOC/voc2012'
           '/VOCtrainval_11-May-2012.tar')
    sha1 = '4e443f8a2eca6b1dac8a6c57641b67dd40621a49'
    fname = gluon.utils.download(url, data_dir, sha1_hash=sha1)
    with tarfile.open(fname, 'r') as f:
        f.extractall(data_dir)
    return voc_dir

def read_voc_images(root='../data/VOCdevkit/VOC2012', is_train=True):
    """Read VOC images."""
    txt_fname = '%s/ImageSets/Segmentation/%s' % (root, 'train.txt' if is_train else 'val.txt')
    with open(txt_fname, 'r') as f:
        images = f.read().split()
    features, labels = [None] * len(images), [None] * len(images)
    for i, fname in enumerate(images):
        features[i] = image.imread('%s/JPEGImages/%s.jpg' % (root, fname))
        labels[i] = image.imread('%s/SegmentationClass/%s.png' % (root, fname))
    return features, labels

def voc_label_indices(colormap, colormap2label):
    """Assign label indices for Pascal VOC2012 Dataset."""
    colormap = colormap.astype('int32')
    idx = ((colormap[:, :, 0] * 256 + colormap[:, :, 1]) * 256
           + colormap[:, :, 2])
    return colormap2label[idx]


def voc_rand_crop(feature, label, height, width):
    """Random cropping for images of the Pascal VOC2012 Dataset."""
    feature, rect = image.random_crop(feature, (width, height))
    label = image.fixed_crop(label, *rect)
    return feature, label

VOC_CLASSES = ['background', 'aeroplane', 'bicycle', 'bird', 'boat',
               'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
               'diningtable', 'dog', 'horse', 'motorbike', 'person',
               'potted plant', 'sheep', 'sofa', 'train', 'tv/monitor']


VOC_COLORMAP = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
                [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
                [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
                [0, 64, 128]]

class VOCSegDataset(gdata.Dataset):
    """The Pascal VOC2012 Dataset."""

    def __init__(self, is_train, crop_size, voc_dir, colormap2label):
        self.rgb_mean = nd.array([0.485, 0.456, 0.406])
        self.rgb_std = nd.array([0.229, 0.224, 0.225])
        self.crop_size = crop_size
        data, labels = read_voc_images(root=voc_dir, is_train=is_train)
        self.data = [self.normalize_image(im) for im in self.filter(data)]
        self.labels = self.filter(labels)
        self.colormap2label = colormap2label
        print('read ' + str(len(self.data)) + ' examples')

    def normalize_image(self, data):
        return (data.astype('float32') / 255 - self.rgb_mean) / self.rgb_std

    def filter(self, images):
        return [im for im in images if (
            im.shape[0] >= self.crop_size[0] and
            im.shape[1] >= self.crop_size[1])]

    def __getitem__(self, idx):
        data, labels = voc_rand_crop(self.data[idx], self.labels[idx],
                                     *self.crop_size)
        return (data.transpose((2, 0, 1)),
                voc_label_indices(labels, self.colormap2label))

    def __len__(self):
        return len(self.data)

def load_data_pascal_voc(batch_size, crop=None, num_workers=None):
    '''
    Load Pascal VOC dataset
    '''
    crop_size, batch_size, colormap2label = crop, batch_size, nd.zeros(256 ** 3)
    for i, cm in enumerate(VOC_COLORMAP):
        colormap2label[(cm[0] * 256 + cm[1]) * 256 + cm[2]] = i

    if os.path.isdir('../data'):
        voc_dir = '../data/VOCdevkit/VOC2012'
    else:
        os.mkdir('../data')
        voc_dir = download_voc_pascal(data_dir='../data')

    if num_workers is None:
        num_workers = 0 if sys.platform.startswith('win32') else 4

    train_iter = gdata.DataLoader(
        VOCSegDataset(True, crop_size, voc_dir, colormap2label), batch_size,
        shuffle=True, last_batch='discard', num_workers=num_workers)
    test_iter = gdata.DataLoader(
        VOCSegDataset(False, crop_size, voc_dir, colormap2label), batch_size,
        last_batch='discard', num_workers=num_workers)
    return train_iter, test_iter

def main():
    # Parse CLI arguments
    opt = parse_args()
    batch_size = opt.batch_size
    num_epochs = opt.num_epochs
    input_size = opt.input_size
    lr = opt.lr
    num_workers = opt.num_workers
    # Set context
    if mx.context.num_gpus() > 0:
        context = mx.gpu(0)
    else:
        context = mx.cpu()
    # Get FCN model
    num_classes = 21
    net = FCN(num_classes, ctx=context)
    # Load train and validation data
    train_data, test_data = load_data_pascal_voc(batch_size, crop=input_size,
                                                 num_workers=num_workers)
    # Define loss and evaluation metrics
    loss = gluon.loss.SoftmaxCrossEntropyLoss(axis=1)
    acc = mx.metric.Accuracy()
    # Hybridize and initialize net
    net.hybridize()
    net[-1].initialize(init.Constant(bilinear_kernel(num_classes, num_classes, 64)), ctx=context)
    net[-2].initialize(init=init.Xavier(), ctx=context)
    # Define trainer
    trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': lr, 'wd': 1e-3})
    # Define estimator
    est = estimator.Estimator(net=net,
                              loss=loss,
                              metrics=acc,
                              trainers=trainer,
                              context=context)
    # Call fit() to begin training
    est.fit(train_data=train_data,
            val_data=test_data,
            epochs=num_epochs,
            batch_size=batch_size,
            event_handlers=[event_handler.LoggingHandler(est, 'fcn_log', 'fcn_log')])


if __name__ == '__main__':
    main()
