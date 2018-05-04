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

import os
import random

import numpy as np

import mxnet as mx
from mxnet import nd

def transform(data, target_wd, target_ht, is_train, box):
    """Crop and normnalize an image nd array."""
    if box is not None:
        x, y, w, h = box
        data = data[y:min(y+h, data.shape[0]), x:min(x+w, data.shape[1])]

    # Resize to target_wd * target_ht.
    data = mx.image.imresize(data, target_wd, target_ht)

    # Normalize in the same way as the pre-trained model.
    data = data.astype(np.float32) / 255.0
    data = (data - mx.nd.array([0.485, 0.456, 0.406])) / mx.nd.array([0.229, 0.224, 0.225])

    if is_train:
        if random.random() < 0.5:
            data = nd.flip(data, axis=1)
        data, _ = mx.image.random_crop(data, (224, 224))
    else:
        data, _ = mx.image.center_crop(data, (224, 224))

    # Transpose from (target_wd, target_ht, 3)
    # to (3, target_wd, target_ht).
    data = nd.transpose(data, (2, 0, 1))

    # If image is greyscale, repeat 3 times to get RGB image.
    if data.shape[0] == 1:
        data = nd.tile(data, (3, 1, 1))
    return data.reshape((1,) + data.shape)


class CUB200Iter(mx.io.DataIter):
    """Iterator for the CUB200-2011 dataset.
    Parameters
    ----------
    data_path : str,
        The path to dataset directory.
    batch_k : int,
        Number of images per class in a batch.
    batch_size : int,
        Batch size.
    batch_size : tupple,
        Data shape. E.g. (3, 224, 224).
    is_train : bool,
        Training data or testig data. Training batches are randomly sampled.
        Testing batches are loaded sequentially until reaching the end.
    """
    def __init__(self, data_path, batch_k, batch_size, data_shape, is_train):
        super(CUB200Iter, self).__init__(batch_size)
        self.data_shape = (batch_size,) + data_shape
        self.batch_size = batch_size
        self.provide_data = [('data', self.data_shape)]
        self.batch_k = batch_k
        self.is_train = is_train

        self.train_image_files = [[] for _ in range(100)]
        self.test_image_files = []
        self.test_labels = []
        self.boxes = {}
        self.test_count = 0

        with open(os.path.join(data_path, 'images.txt'), 'r') as f_img, \
             open(os.path.join(data_path, 'image_class_labels.txt'), 'r') as f_label, \
             open(os.path.join(data_path, 'bounding_boxes.txt'), 'r') as f_box:
            for line_img, line_label, line_box in zip(f_img, f_label, f_box):
                fname = os.path.join(data_path, 'images', line_img.strip().split()[-1])
                label = int(line_label.strip().split()[-1]) - 1
                box = [int(float(v)) for v in line_box.split()[-4:]]
                self.boxes[fname] = box

                # Following "Deep Metric Learning via Lifted Structured Feature Embedding" paper,
                # we use the first 100 classes for training, and the remaining for testing.
                if label < 100:
                    self.train_image_files[label].append(fname)
                else:
                    self.test_labels.append(label)
                    self.test_image_files.append(fname)

        self.n_test = len(self.test_image_files)

    def get_image(self, img, is_train):
        """Load and transform an image."""
        img_arr = mx.image.imread(img)
        img_arr = transform(img_arr, 256, 256, is_train, self.boxes[img])
        return img_arr

    def sample_train_batch(self):
        """Sample a training batch (data and label)."""
        batch = []
        labels = []
        num_groups = self.batch_size // self.batch_k

        # For CUB200, we use the first 100 classes for training.
        sampled_classes = np.random.choice(100, num_groups, replace=False)
        for i in range(num_groups):
            img_fnames = np.random.choice(self.train_image_files[sampled_classes[i]],
                                          self.batch_k, replace=False)
            batch += [self.get_image(img_fname, is_train=True) for img_fname in img_fnames]
            labels += [sampled_classes[i] for _ in range(self.batch_k)]

        return nd.concatenate(batch, axis=0), labels

    def get_test_batch(self):
        """Sample a testing batch (data and label)."""

        batch_size = self.batch_size
        batch = [self.get_image(self.test_image_files[(self.test_count*batch_size + i)
                                                      % len(self.test_image_files)],
                                is_train=False) for i in range(batch_size)]
        labels = [self.test_labels[(self.test_count*batch_size + i)
                                   % len(self.test_image_files)] for i in range(batch_size)]
        return nd.concatenate(batch, axis=0), labels

    def reset(self):
        """Reset an iterator."""
        self.test_count = 0

    def next(self):
        """Return a batch."""
        if self.is_train:
            data, labels = self.sample_train_batch()
        else:
            if self.test_count * self.batch_size < len(self.test_image_files):
                data, labels = self.get_test_batch()
                self.test_count += 1
            else:
                self.test_count = 0
                raise StopIteration
        return mx.io.DataBatch(data=[data], label=[labels])

def cub200_iterator(data_path, batch_k, batch_size, data_shape):
    """Return training and testing iterator for the CUB200-2011 dataset."""
    return (CUB200Iter(data_path, batch_k, batch_size, data_shape, is_train=True),
            CUB200Iter(data_path, batch_k, batch_size, data_shape, is_train=False))
