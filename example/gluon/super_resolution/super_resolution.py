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
import math
import os
import shutil
import sys
import zipfile
from os import path

import numpy as np

import mxnet as mx
from mxnet import gluon, autograd as ag
from mxnet.gluon import nn
from mxnet.image import CenterCropAug, ResizeAug
from mxnet.io import PrefetchingIter
from mxnet.test_utils import download

this_dir = path.abspath(path.dirname(__file__))
sys.path.append(path.join(this_dir, path.pardir))

from data import ImagePairIter


# CLI
parser = argparse.ArgumentParser(description='Super-resolution using an efficient sub-pixel convolution neural network.')
parser.add_argument('--upscale_factor', type=int, default=3, help="super resolution upscale factor. default is 3.")
parser.add_argument('--batch_size', type=int, default=4, help='training batch size, per device. default is 4.')
parser.add_argument('--test_batch_size', type=int, default=100, help='test batch size')
parser.add_argument('--epochs', type=int, default=30, help='number of training epochs')
parser.add_argument('--lr', type=float, default=0.001, help='learning Rate. default is 0.001.')
parser.add_argument('--use-gpu', action='store_true', help='whether to use GPU.')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument('--resolve_img', type=str, help='input image to use')
opt = parser.parse_args()

print(opt)

upscale_factor = opt.upscale_factor
batch_size, test_batch_size = opt.batch_size, opt.test_batch_size
color_flag = 0

# Get data from https://github.com/BIDS/BSDS500/
# The BSDS500 Dataset is copyright Berkeley Computer Vision Group
# For more details, see https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/resources.html#bsds500
datasets_dir = path.expanduser(path.join("~", ".mxnet", "datasets"))
datasets_tmpdir = path.join(datasets_dir, "tmp")
dataset_url = "https://github.com/BIDS/BSDS500/archive/master.zip"
data_dir = path.expanduser(path.join(datasets_dir, "BSDS500"))
tmp_dir = path.join(data_dir, "tmp")

def get_dataset(prefetch=False):
    """Download the BSDS500 dataset and return train and test iters."""

    if path.exists(data_dir):
        print(
            "Directory {} already exists, skipping.\n"
            "To force download and extraction, delete the directory and re-run."
            "".format(data_dir),
            file=sys.stderr,
        )
    else:
        print("Downloading dataset...", file=sys.stderr)
        downloaded_file = download(dataset_url, dirname=datasets_tmpdir)
        print("done", file=sys.stderr)

        print("Extracting files...", end="", file=sys.stderr)
        os.makedirs(data_dir)
        os.makedirs(tmp_dir)
        with zipfile.ZipFile(downloaded_file) as archive:
            archive.extractall(tmp_dir)
        shutil.rmtree(datasets_tmpdir)

        shutil.copytree(
            path.join(tmp_dir, "BSDS500-master", "BSDS500", "data", "images"),
            path.join(data_dir, "images"),
        )
        shutil.copytree(
            path.join(tmp_dir, "BSDS500-master", "BSDS500", "data", "groundTruth"),
            path.join(data_dir, "groundTruth"),
        )
        shutil.rmtree(tmp_dir)
        print("done", file=sys.stderr)

    crop_size = 256
    crop_size -= crop_size % upscale_factor
    input_crop_size = crop_size // upscale_factor

    input_transform = [CenterCropAug((crop_size, crop_size)), ResizeAug(input_crop_size)]
    target_transform = [CenterCropAug((crop_size, crop_size))]

    iters = (
        ImagePairIter(
            path.join(data_dir, "images", "train"),
            (input_crop_size, input_crop_size),
            (crop_size, crop_size),
            batch_size,
            color_flag,
            input_transform,
            target_transform,
        ),
        ImagePairIter(
            path.join(data_dir, "images", "test"),
            (input_crop_size, input_crop_size),
            (crop_size, crop_size),
            test_batch_size,
            color_flag,
            input_transform,
            target_transform,
        ),
    )

    return [PrefetchingIter(i) for i in iters] if prefetch else iters

train_data, val_data = get_dataset()

mx.np.random.seed(opt.seed)
device = [mx.gpu(0)] if opt.use_gpu else [mx.cpu()]


class SuperResolutionNet(gluon.HybridBlock):
    def __init__(self, upscale_factor):
        super(SuperResolutionNet, self).__init__()
        self.conv1 = nn.Conv2D(64, (5, 5), strides=(1, 1), padding=(2, 2), activation='relu')
        self.conv2 = nn.Conv2D(64, (3, 3), strides=(1, 1), padding=(1, 1), activation='relu')
        self.conv3 = nn.Conv2D(32, (3, 3), strides=(1, 1), padding=(1, 1), activation='relu')
        self.conv4 = nn.Conv2D(upscale_factor ** 2, (3, 3), strides=(1, 1), padding=(1, 1))
        self.pxshuf = nn.PixelShuffle2D(upscale_factor)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.pxshuf(x)
        return x

net = SuperResolutionNet(upscale_factor)
metric = mx.gluon.metric.MSE()

def test(device):
    val_data.reset()
    avg_psnr = 0
    batches = 0
    for batch in val_data:
        batches += 1
        data = gluon.utils.split_and_load(batch.data[0], device_list=device, batch_axis=0)
        label = gluon.utils.split_and_load(batch.label[0], device_list=device, batch_axis=0)
        outputs = []
        for x in data:
            outputs.append(net(x))
        metric.update(label, outputs)
        avg_psnr += 10 * math.log10(1/metric.get()[1])
        metric.reset()
    avg_psnr /= batches
    print(f'validation avg psnr: {avg_psnr}')


def train(epoch, device):
    if isinstance(device, mx.Device):
        device = [device]
    net.initialize(mx.init.Orthogonal(), device=device)
    # re-initialize conv4's weight to be Orthogonal
    net.conv4.initialize(mx.init.Orthogonal(scale=1), force_reinit=True, device=device)
    trainer = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate': opt.lr})
    loss = gluon.loss.L2Loss()

    for i in range(epoch):
        train_data.reset()
        for batch in train_data:
            data = gluon.utils.split_and_load(batch.data[0], device_list=device, batch_axis=0)
            label = gluon.utils.split_and_load(batch.label[0], device_list=device, batch_axis=0)
            outputs = []
            with ag.record():
                for x, y in zip(data, label):
                    z = net(x)
                    L = loss(z, y)
                    L.backward()
                    outputs.append(z)
            trainer.step(batch.data[0].shape[0])
            metric.update(label, outputs)

        name, acc = metric.get()
        metric.reset()
        print(f'training mse at epoch {i}: {name}={acc}')
        test(device)

    net.save_parameters(path.join(this_dir, 'superres.params'))

def resolve(device):
    from PIL import Image

    if isinstance(device, list):
        device = [device[0]]

    img_basename = path.splitext(path.basename(opt.resolve_img))[0]
    img_dirname = path.dirname(opt.resolve_img)

    net.load_parameters(path.join(this_dir, 'superres.params'), device=device)
    img = Image.open(opt.resolve_img).convert('YCbCr')
    y, cb, cr = img.split()
    data = mx.np.expand_dims(mx.np.expand_dims(mx.np.array(y), axis=0), axis=0)
    out_img_y = mx.np.reshape(net(data), shape=(-3, -2)).asnumpy()
    out_img_y = out_img_y.clip(0, 255)
    out_img_y = Image.fromarray(np.uint8(out_img_y[0]), mode='L')

    out_img_cb = cb.resize(out_img_y.size, Image.BICUBIC)
    out_img_cr = cr.resize(out_img_y.size, Image.BICUBIC)
    out_img = Image.merge('YCbCr', [out_img_y, out_img_cb, out_img_cr]).convert('RGB')

    out_img.save(path.join(img_dirname, '{}-resolved.png'.format(img_basename)))

if opt.resolve_img:
    resolve(device)
else:
    train(opt.epochs, device)
