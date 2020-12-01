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
import numpy as np
from mxnet.test_utils import *
import sys
import shutil
import tempfile
import unittest
import pytest

def _get_data(url, dirname):
    import os, tarfile
    download(url, dirname=dirname, overwrite=False)
    fname = os.path.join(dirname, url.split('/')[-1])
    tar = tarfile.open(fname)
    source_images = [os.path.join(dirname, x.name) for x in tar.getmembers() if x.isfile()]
    if len(source_images) < 1 or not os.path.isfile(source_images[0]):
        # skip extracting if exists
        tar.extractall(path=dirname)
    tar.close()
    return source_images

def _generate_objects():
    num = np.random.randint(1, 10)
    xy = np.random.rand(num, 2)
    wh = np.random.rand(num, 2) / 2
    left = (xy[:, 0] - wh[:, 0])[:, np.newaxis]
    right = (xy[:, 0] + wh[:, 0])[:, np.newaxis]
    top = (xy[:, 1] - wh[:, 1])[:, np.newaxis]
    bot = (xy[:, 1] + wh[:, 1])[:, np.newaxis]
    boxes = np.maximum(0., np.minimum(1., np.hstack((left, top, right, bot))))
    cid = np.random.randint(0, 20, size=num)
    label = np.hstack((cid[:, np.newaxis], boxes)).ravel().tolist()
    return [2, 5] + label


class TestImage(unittest.TestCase):
    IMAGES_URL = "https://repo.mxnet.io/gluon/dataset/test/test_images-9cebe48a.tar.gz"

    def setUp(self):
        self.IMAGES_DIR = tempfile.mkdtemp()
        self.IMAGES = _get_data(self.IMAGES_URL, self.IMAGES_DIR)
        print("Loaded {} images".format(len(self.IMAGES)))

    def tearDown(self):
        if self.IMAGES_DIR:
            print("cleanup {}".format(self.IMAGES_DIR))
            shutil.rmtree(self.IMAGES_DIR)

    @use_np
    def test_imageiter(self):
        im_list = [[np.random.randint(0, 5), x] for x in self.IMAGES]
        fname = './data/test_numpy_imageiter.lst'
        file_list = ['\t'.join([str(k), str(np.random.randint(0, 5)), x])
                        for k, x in enumerate(self.IMAGES)]
        with open(fname, 'w') as f:
            for line in file_list:
                f.write(line + '\n')

        test_list = ['imglist', 'path_imglist']
        for dtype in ['int32', 'float32', 'int64', 'float64']:
            for test in test_list:
                imglist = im_list if test == 'imglist' else None
                path_imglist = fname if test == 'path_imglist' else None
                imageiter_list = [
                    mx.gluon.contrib.data.vision.ImageDataLoader(2, (3, 224, 224), imglist=imglist,
                        path_imglist=path_imglist, path_root='', dtype=dtype),
                    mx.gluon.contrib.data.vision.ImageDataLoader(3, (3, 224, 224), imglist=imglist,
                        path_imglist=path_imglist, path_root='', dtype=dtype, last_batch='discard'),
                    mx.gluon.contrib.data.vision.ImageDataLoader(3, (3, 224, 224), imglist=imglist,
                        path_imglist=path_imglist, path_root='', dtype=dtype, last_batch='keep'),
                    mx.gluon.contrib.data.vision.ImageDataLoader(3, (3, 224, 224), imglist=imglist,
                        path_imglist=path_imglist, path_root='', dtype=dtype, last_batch='rollover'),
                    mx.gluon.contrib.data.vision.ImageDataLoader(3, (3, 224, 224), imglist=imglist, shuffle=True,
                        path_imglist=path_imglist, path_root='', dtype=dtype, last_batch='keep',
                        rand_crop=1, rand_gray=0.1, rand_mirror=True)
                ]
                for it in imageiter_list:
                    for batch in it:
                        pass

    @use_np
    def test_image_bbox_iter(self):
        im_list = [_generate_objects() + [x] for x in self.IMAGES]
        det_iter = mx.gluon.contrib.data.vision.ImageBboxDataLoader(2, (3, 300, 300), imglist=im_list, path_root='')
        for _ in range(3):
            for _ in det_iter:
                pass
        val_iter = mx.gluon.contrib.data.vision.ImageBboxDataLoader(2, (3, 300, 300), imglist=im_list, path_root='')

        # test batch_size is not divisible by number of images
        det_iter = mx.gluon.contrib.data.vision.ImageBboxDataLoader(4, (3, 300, 300), imglist=im_list, path_root='')
        for _ in det_iter:
            pass

        # test file list with last batch handle
        fname = './data/test_numpy_imagedetiter.lst'
        im_list = [[k] + _generate_objects() + [x] for k, x in enumerate(self.IMAGES)]
        with open(fname, 'w') as f:
            for line in im_list:
                line = '\t'.join([str(k) for k in line])
                f.write(line + '\n')

        imageiter_list = [
            mx.gluon.contrib.data.vision.ImageBboxDataLoader(2, (3, 400, 400),
                path_imglist=fname, path_root=''),
            mx.gluon.contrib.data.vision.ImageBboxDataLoader(3, (3, 400, 400),
                path_imglist=fname, path_root='', last_batch='discard'),
            mx.gluon.contrib.data.vision.ImageBboxDataLoader(3, (3, 400, 400),
                path_imglist=fname, path_root='', last_batch='keep'),
            mx.gluon.contrib.data.vision.ImageBboxDataLoader(3, (3, 400, 400),
                path_imglist=fname, path_root='', last_batch='rollover'),
            mx.gluon.contrib.data.vision.ImageBboxDataLoader(3, (3, 400, 400), shuffle=True,
                path_imglist=fname, path_root='', last_batch='keep')
        ]

    @use_np
    def test_bbox_augmenters(self):
        # only test if all augmenters will work
        im_list = [_generate_objects() + [x] for x in self.IMAGES]
        det_iter = mx.gluon.contrib.data.vision.ImageBboxDataLoader(2, (3, 300, 300), imglist=im_list, path_root='',
            rand_crop=1, rand_pad=1, rand_gray=0.1, rand_mirror=True, mean=True,
            std=[1.1, 1.03, 1.05], brightness=0.1, contrast=0.1, saturation=0.1,
            pca_noise=0.1, hue=0.1, inter_method=10,
            max_aspect_ratio=5, area_range=(0.1, 4.0),
            max_attempts=50)
        for batch in det_iter:
            assert np.dtype(batch[1].dtype) == np.float32, str(np.dtype(batch[1].dtype)) + ': ' + str(batch[1])
            pass
