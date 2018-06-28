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

# pylint: skip-file
""" data iterator for mnist """
import os
import random
import logging
logging.basicConfig(level=logging.INFO)

import mxnet as mx
from mxnet.test_utils import get_cifar10
from mxnet.gluon.data.vision import ImageFolderDataset
from mxnet.gluon.data import DataLoader
from mxnet.contrib.io import DataLoaderIter

def get_cifar10_iterator(batch_size, data_shape, resize=-1, num_parts=1, part_index=0):
    get_cifar10()

    train = mx.io.ImageRecordIter(
        path_imgrec = "data/cifar/train.rec",
        # mean_img    = "data/cifar/mean.bin",
        resize      = resize,
        data_shape  = data_shape,
        batch_size  = batch_size,
        rand_crop   = True,
        rand_mirror = True,
        num_parts=num_parts,
        part_index=part_index)

    val = mx.io.ImageRecordIter(
        path_imgrec = "data/cifar/test.rec",
        # mean_img    = "data/cifar/mean.bin",
        resize      = resize,
        rand_crop   = False,
        rand_mirror = False,
        data_shape  = data_shape,
        batch_size  = batch_size,
        num_parts=num_parts,
        part_index=part_index)

    return train, val

def get_imagenet_transforms(data_shape=224, dtype='float32'):
    def train_transform(image, label):
        image, _ = mx.image.random_size_crop(image, (data_shape, data_shape), 0.08, (3/4., 4/3.))
        image = mx.nd.image.random_flip_left_right(image)
        image = mx.nd.image.to_tensor(image)
        image = mx.nd.image.normalize(image, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        return mx.nd.cast(image, dtype), label

    def val_transform(image, label):
        image = mx.image.resize_short(image, data_shape + 32)
        image, _ = mx.image.center_crop(image, (data_shape, data_shape))
        image = mx.nd.image.to_tensor(image)
        image = mx.nd.image.normalize(image, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        return mx.nd.cast(image, dtype), label
    return train_transform, val_transform

def get_imagenet_iterator(root, batch_size, num_workers, data_shape=224, dtype='float32'):
    """Dataset loader with preprocessing."""
    train_dir = os.path.join(root, 'train')
    train_transform, val_transform = get_imagenet_transforms(data_shape, dtype)
    logging.info("Loading image folder %s, this may take a bit long...", train_dir)
    train_dataset = ImageFolderDataset(train_dir, transform=train_transform)
    train_data = DataLoader(train_dataset, batch_size, shuffle=True,
                            last_batch='discard', num_workers=num_workers)
    val_dir = os.path.join(root, 'val')
    if not os.path.isdir(os.path.expanduser(os.path.join(root, 'val', 'n01440764'))):
        user_warning = 'Make sure validation images are stored in one subdir per category, a helper script is available at https://git.io/vNQv1'
        raise ValueError(user_warning)
    logging.info("Loading image folder %s, this may take a bit long...", val_dir)
    val_dataset = ImageFolderDataset(val_dir, transform=val_transform)
    val_data = DataLoader(val_dataset, batch_size, last_batch='keep', num_workers=num_workers)
    return DataLoaderIter(train_data, dtype), DataLoaderIter(val_data, dtype)


class DummyIter(mx.io.DataIter):
    def __init__(self, batch_size, data_shape, batches = 100):
        super(DummyIter, self).__init__(batch_size)
        self.data_shape = (batch_size,) + data_shape
        self.label_shape = (batch_size,)
        self.provide_data = [('data', self.data_shape)]
        self.provide_label = [('softmax_label', self.label_shape)]
        self.batch = mx.io.DataBatch(data=[mx.nd.zeros(self.data_shape)],
                                     label=[mx.nd.zeros(self.label_shape)])
        self._batches = 0
        self.batches = batches

    def next(self):
        if self._batches < self.batches:
            self._batches += 1
            return self.batch
        else:
            self._batches = 0
            raise StopIteration

def dummy_iterator(batch_size, data_shape):
    return DummyIter(batch_size, data_shape), DummyIter(batch_size, data_shape)

class ImagePairIter(mx.io.DataIter):
    def __init__(self, path, data_shape, label_shape, batch_size=64, flag=0, input_aug=None, target_aug=None):
        super(ImagePairIter, self).__init__(batch_size)
        self.data_shape = (batch_size,) + data_shape
        self.label_shape = (batch_size,) + label_shape
        self.input_aug = input_aug
        self.target_aug = target_aug
        self.provide_data = [('data', self.data_shape)]
        self.provide_label = [('label', self.label_shape)]
        is_image_file = lambda fn: any(fn.endswith(ext) for ext in [".png", ".jpg", ".jpeg"])
        self.filenames = [os.path.join(path, x) for x in os.listdir(path) if is_image_file(x)]
        self.count = 0
        self.flag = flag
        random.shuffle(self.filenames)

    def next(self):
        from PIL import Image
        if self.count + self.batch_size <= len(self.filenames):
            data = []
            label = []
            for i in range(self.batch_size):
                fn = self.filenames[self.count]
                self.count += 1
                image = Image.open(fn).convert('YCbCr').split()[0]
                if image.size[0] > image.size[1]:
                    image = image.transpose(Image.TRANSPOSE)
                image = mx.nd.expand_dims(mx.nd.array(image), axis=2)
                target = image.copy()
                for aug in self.input_aug:
                    image = aug(image)
                for aug in self.target_aug:
                    target = aug(target)
                data.append(image)
                label.append(target)

            data = mx.nd.concat(*[mx.nd.expand_dims(d, axis=0) for d in data], dim=0)
            label = mx.nd.concat(*[mx.nd.expand_dims(d, axis=0) for d in label], dim=0)
            data = [mx.nd.transpose(data, axes=(0, 3, 1, 2)).astype('float32')/255]
            label = [mx.nd.transpose(label, axes=(0, 3, 1, 2)).astype('float32')/255]

            return mx.io.DataBatch(data=data, label=label)
        else:
            raise StopIteration

    def reset(self):
        self.count = 0
        random.shuffle(self.filenames)
