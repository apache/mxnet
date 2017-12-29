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
import argparse, tarfile
import math
import os
import numpy as np

import mxnet as mx
import mxnet.ndarray as F
from mxnet import gluon
from mxnet.gluon import nn
from mxnet import autograd as ag
from mxnet.test_utils import download
from mxnet.image import CenterCropAug, ResizeAug
from mxnet.io import PrefetchingIter

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

# get data
dataset_path = "dataset"
dataset_url = "http://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/BSDS300-images.tgz"
def get_dataset(prefetch=False):
    image_path = os.path.join(dataset_path, "BSDS300/images")

    if not os.path.exists(image_path):
        os.makedirs(dataset_path)
        file_name = download(dataset_url)
        with tarfile.open(file_name) as tar:
            for item in tar:
                tar.extract(item, dataset_path)
        os.remove(file_name)

    crop_size = 256
    crop_size -= crop_size % upscale_factor
    input_crop_size = crop_size // upscale_factor

    input_transform = [CenterCropAug((crop_size, crop_size)), ResizeAug(input_crop_size)]
    target_transform = [CenterCropAug((crop_size, crop_size))]

    iters = (ImagePairIter(os.path.join(image_path, "train"),
                           (input_crop_size, input_crop_size),
                           (crop_size, crop_size),
                           batch_size, color_flag, input_transform, target_transform),
             ImagePairIter(os.path.join(image_path, "test"),
                           (input_crop_size, input_crop_size),
                           (crop_size, crop_size),
                           test_batch_size, color_flag,
                           input_transform, target_transform))

    return [PrefetchingIter(i) for i in iters] if prefetch else iters

train_data, val_data = get_dataset()

mx.random.seed(opt.seed)
ctx = [mx.gpu(0)] if opt.use_gpu else [mx.cpu()]


# define model
def _rearrange(raw, F, upscale_factor):
    # (N, C * r^2, H, W) -> (N, C, r^2, H, W)
    splitted = F.reshape(raw, shape=(0, -4, -1, upscale_factor**2, 0, 0))
    # (N, C, r^2, H, W) -> (N, C, r, r, H, W)
    unflatten = F.reshape(splitted, shape=(0, 0, -4, upscale_factor, upscale_factor, 0, 0))
    # (N, C, r, r, H, W) -> (N, C, H, r, W, r)
    swapped = F.transpose(unflatten, axes=(0, 1, 4, 2, 5, 3))
    # (N, C, H, r, W, r) -> (N, C, H*r, W*r)
    return F.reshape(swapped, shape=(0, 0, -3, -3))


class SuperResolutionNet(gluon.Block):
    def __init__(self, upscale_factor):
        super(SuperResolutionNet, self).__init__()
        with self.name_scope():
            self.conv1 = nn.Conv2D(64, (5, 5), strides=(1, 1), padding=(2, 2))
            self.conv2 = nn.Conv2D(64, (3, 3), strides=(1, 1), padding=(1, 1))
            self.conv3 = nn.Conv2D(32, (3, 3), strides=(1, 1), padding=(1, 1))
            self.conv4 = nn.Conv2D(upscale_factor ** 2, (3, 3), strides=(1, 1), padding=(1, 1))
        self.upscale_factor = upscale_factor

    def forward(self, x):
        x = F.Activation(self.conv1(x), act_type='relu')
        x = F.Activation(self.conv2(x), act_type='relu')
        x = F.Activation(self.conv3(x), act_type='relu')
        return _rearrange(self.conv4(x), F, self.upscale_factor)

net = SuperResolutionNet(upscale_factor)
metric = mx.metric.MSE()

def test(ctx):
    val_data.reset()
    avg_psnr = 0
    batches = 0
    for batch in val_data:
        batches += 1
        data = gluon.utils.split_and_load(batch.data[0], ctx_list=ctx, batch_axis=0)
        label = gluon.utils.split_and_load(batch.label[0], ctx_list=ctx, batch_axis=0)
        outputs = []
        for x in data:
            outputs.append(net(x))
        metric.update(label, outputs)
        avg_psnr += 10 * math.log10(1/metric.get()[1])
        metric.reset()
    avg_psnr /= batches
    print('validation avg psnr: %f'%avg_psnr)


def train(epoch, ctx):
    if isinstance(ctx, mx.Context):
        ctx = [ctx]
    net.initialize(mx.init.Orthogonal(), ctx=ctx)
    # re-initialize conv4's weight to be Orthogonal
    net.conv4.collect_params().initialize(mx.init.Orthogonal(scale=1), force_reinit=True, ctx=ctx)
    trainer = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate': opt.lr})
    loss = gluon.loss.L2Loss()

    for i in range(epoch):
        train_data.reset()
        for batch in train_data:
            data = gluon.utils.split_and_load(batch.data[0], ctx_list=ctx, batch_axis=0)
            label = gluon.utils.split_and_load(batch.label[0], ctx_list=ctx, batch_axis=0)
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
        print('training mse at epoch %d: %s=%f'%(i, name, acc))
        test(ctx)

    net.save_params('superres.params')

def resolve(ctx):
    from PIL import Image
    if isinstance(ctx, list):
        ctx = [ctx[0]]
    net.load_params('superres.params', ctx=ctx)
    img = Image.open(opt.resolve_img).convert('YCbCr')
    y, cb, cr = img.split()
    data = mx.nd.expand_dims(mx.nd.expand_dims(mx.nd.array(y), axis=0), axis=0)
    out_img_y = mx.nd.reshape(net(data), shape=(-3, -2)).asnumpy()
    out_img_y = out_img_y.clip(0, 255)
    out_img_y = Image.fromarray(np.uint8(out_img_y[0]), mode='L')

    out_img_cb = cb.resize(out_img_y.size, Image.BICUBIC)
    out_img_cr = cr.resize(out_img_y.size, Image.BICUBIC)
    out_img = Image.merge('YCbCr', [out_img_y, out_img_cb, out_img_cr]).convert('RGB')

    out_img.save('resolved.png')

if opt.resolve_img:
    resolve(ctx)
else:
    train(opt.epochs, ctx)
