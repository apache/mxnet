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
""" file iterator for pasval voc 2012"""

import mxnet as mx
import numpy as np
import random
import sys, os
from mxnet.io import DataIter

class FileIter(DataIter):
    """FileIter object in fcn-xs example. Taking a file list file to get dataiter.
    in this example, we use the whole image training for fcn-xs, that is to say
    we do not need resize/crop the image to the same size, so the batch_size is
    set to 1 here
    Parameters
    ----------
    batch_size : int
        Number of examples per batch.
    root_dir : string
        the root dir of image/label lie in
    flist_name : string
        the list file of iamge and label, every line owns the form:
        index \t image_data_path \t image_label_path
    cut_off_size : int
        if the maximal size of one image is larger than cut_off_size, then it will
        crop the image with the minimal size of that image
    data_name : string
        the data name used in symbol data(default data name)
    label_name : string
        the label name used in symbol softmax_label(default label name)
    data_shape : tuple
        Data shape in (channels, height, width) format.
        For now, only RGB image with 3 channels is supported.    
    part_index : int
        Partition index.
    num_parts : int
        Total number of partitions.  
    shuffle : bool
        Shuffle the data rank.    
    """
    def __init__(self, batch_size, root_dir, flist_name,
                 rgb_mean = (117, 117, 117),
                 cut_off_size = None,
                 data_name = "data",
                 label_name = "softmax_label",
                 data_shape = (3, 360, 480),
                 part_index = 0,
                 num_parts = 1,
                 shuffle = True):
        super(FileIter, self).__init__()
        self.root_dir = root_dir
        self.flist_name = os.path.join(self.root_dir, flist_name)
        self.mean = np.array(rgb_mean)  # (R, G, B)
        self.cut_off_size = cut_off_size
        self.data_name = data_name
        self.label_name = label_name

        self.num_data = len(open(self.flist_name, 'r').readlines())
        self.f = open(self.flist_name, 'r')
        self.names = self.f.read().split("\n")[0:-1]
        self.cursor = part_index
        self.num_parts = num_parts
        self.part_index = part_index
        self.batch_size = batch_size
        self.data_shape = data_shape
        self.shuffle = shuffle
        self.data, self.label = self._read()

    def _read(self):
        """get two list, each list contains two elements: name and nd.array value"""
        #img_name= self.f.readline().strip('\n').split("\t")[0].split(' ')
        img_name = self.names[self.cursor + self.num_parts].split(' ')
        data = {}
        label = {}
        data[self.data_name], label[self.label_name] = self._read_img(img_name[0], img_name[1])
        return list(data.items()), list(label.items())

    def _read_img(self, img_name, label_name):
        img = mx.image.imread(os.path.join(self.root_dir, img_name))
        label = mx.image.imread(os.path.join(self.root_dir, label_name), flag=0)
        img = img.astype('float32')
        label = label.astype('float32')
        if self.cut_off_size is not None:
            max_hw = max(img.shape[0], img.shape[1])
            min_hw = min(img.shape[0], img.shape[1])
            if min_hw > self.cut_off_size:
                rand_start_max = int(np.random.uniform(0, max_hw - self.cut_off_size - 1))
                rand_start_min = int(np.random.uniform(0, min_hw - self.cut_off_size - 1))
                if img.shape[0] == max_hw :
                    img = img[rand_start_max : rand_start_max + self.cut_off_size, rand_start_min : rand_start_min + self.cut_off_size]
                    label = label[rand_start_max : rand_start_max + self.cut_off_size, rand_start_min : rand_start_min + self.cut_off_size]
                else :
                    img = img[rand_start_min : rand_start_min + self.cut_off_size, rand_start_max : rand_start_max + self.cut_off_size]
                    label = label[rand_start_min : rand_start_min + self.cut_off_size, rand_start_max : rand_start_max + self.cut_off_size]
            elif max_hw > self.cut_off_size:
                rand_start = int(np.random.uniform(0, max_hw - min_hw - 1))
                if img.shape[0] == max_hw :
                    img = img[rand_start : rand_start + min_hw, :]
                    label = label[rand_start : rand_start + min_hw, :]
                else :
                    img = img[:, rand_start : rand_start + min_hw]
                    label = label[:, rand_start : rand_start + min_hw]
        reshaped_mean = self.mean.reshape((1, 1, 3))
        img = img - reshaped_mean
        img = mx.nd.moveaxis(img, 2, 0)
        img = mx.nd.expand_dims(img, axis=0)
        label = mx.nd.moveaxis(label, 2, 0)
        return (img, label)

    @property
    def provide_data(self):
        """The name and shape of data provided by this iterator"""
        return [(k, tuple([self.batch_size] + list(v.shape[1:]))) for k, v in self.data]

    @property
    def provide_label(self):
        """The name and shape of label provided by this iterator"""
        return [(k, tuple([self.batch_size] + list(v.shape[1:]))) for k, v in self.label]

    def get_batch_size(self):
        return self.batch_size

    def reset(self):
        self.cursor = self.part_index
        #self.f.close()
        #self.f = open(self.flist_name, 'r')
        if self.shuffle:
            random.shuffle(self.names)

    def iter_next(self):
        self.cursor += self.num_parts
        if(self.cursor < self.num_data-1):
            return True
        else:
            return False

    def next(self):
        """return one dict which contains "data" and "label" """
        c, h, w = self.data_shape
        batch_data = mx.ndarray.empty((self.batch_size, c, h, w))
        batch_label = mx.ndarray.empty((self.batch_size, h, w))
        i = 0
        while i < self.batch_size: 
            if self.iter_next():
                self.data, self.label = self._read()
                batch_data[i][:] = self.data[0][1][0]
                batch_label[i][:] = self.label[0][1][0]
                i += 1
            else:
                raise StopIteration    
        return mx.io.DataBatch([batch_data], [batch_label], pad = self.batch_size - i)
