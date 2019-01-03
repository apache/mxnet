"""
Description : Set DataSet module for lip images
"""
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
import glob
from mxnet import nd
import mxnet.gluon.data.dataset as dataset
from mxnet.gluon.data.vision.datasets import image
from utils.align import Align

class LipsDataset(dataset.Dataset):
    """
    Description : DataSet class for lip images
    """
    def __init__(self, root, align_root, flag=1, transform=None):
        self._root = os.path.expanduser(root)
        self._align_root = align_root
        self._flag = flag
        self._transform = transform
        self._exts = ['.jpg', '.jpeg', '.png']
        self._list_images(self._root)

    def _list_images(self, root):
        """
        Description : generate list for lip images
        """
        self.labels = []
        self.items = []
        folder_path = glob.glob(os.path.join(root, "*", "*"))
        for folder in folder_path:
            filename = glob.glob(os.path.join(folder, "*"))
            if len(filename) != 75:
                continue
            filename.sort()
            label = os.path.split(folder)[-1]
            self.items.append((filename, label))
    def align_generation(self, file_nm, padding=75):
        """
        Description : Align to lip position
        """
        align = Align(self._align_root + '/' + file_nm + '.align')
        return nd.array(align.sentence(padding))
    def __getitem__(self, idx):
        img = list()
        for image_name in self.items[idx][0]:
            tmp_img = image.imread(image_name, self._flag)
            if self._transform is not None:
                tmp_img = self._transform(tmp_img)
            img.append(tmp_img)
        img = nd.stack(*img)
        #print(self.items[idx][0][0])
        label = self.align_generation(self.items[idx][1])
        return img, label

    def __len__(self):
        return len(self.items)
