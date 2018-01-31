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
from common import assertRaises
import shutil
import tempfile
import unittest

from nose.tools import raises

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
    IMAGES_URL = "http://data.mxnet.io/data/test_images.tar.gz"
    IMAGES = []
    IMAGES_DIR = None

    @classmethod
    def setupClass(cls):
        cls.IMAGES_DIR = tempfile.mkdtemp()
        cls.IMAGES = _get_data(cls.IMAGES_URL, cls.IMAGES_DIR)
        print("Loaded {} images".format(len(cls.IMAGES)))

    @classmethod
    def teardownClass(cls):
        if cls.IMAGES_DIR:
            print("cleanup {}".format(cls.IMAGES_DIR))
            shutil.rmtree(cls.IMAGES_DIR)

    @raises(mx.base.MXNetError)
    def test_imread_not_found(self):
        x = mx.img.image.imread("/139810923jadjsajlskd.___adskj/blah.jpg")

    def test_imread_vs_imdecode(self):
        for img in TestImage.IMAGES:
            with open(img, 'rb') as fp:
                str_image = fp.read()
                image = mx.image.imdecode(str_image, to_rgb=0)
                image_read = mx.img.image.imread(img)
                same(image.asnumpy(), image_read.asnumpy())


    def test_imdecode(self):
        try:
            import cv2
        except ImportError:
            return
        for img in TestImage.IMAGES:
            with open(img, 'rb') as fp:
                str_image = fp.read()
                image = mx.image.imdecode(str_image, to_rgb=0)
            cv_image = cv2.imread(img)
            assert_almost_equal(image.asnumpy(), cv_image)

    def test_scale_down(self):
        assert mx.image.scale_down((640, 480), (720, 120)) == (640, 106)
        assert mx.image.scale_down((360, 1000), (480, 500)) == (360, 375)
        assert mx.image.scale_down((300, 400), (0, 0)) == (0, 0)

    def test_resize_short(self):
        try:
            import cv2
        except ImportError:
            return
        for img in TestImage.IMAGES:
            cv_img = cv2.imread(img)
            mx_img = mx.nd.array(cv_img[:, :, (2, 1, 0)])
            h, w, _ = cv_img.shape
            for _ in range(3):
                new_size = np.random.randint(1, 1000)
                if h > w:
                    new_h, new_w = new_size * h / w, new_size
                else:
                    new_h, new_w = new_size, new_size * w / h
                for interp in range(0, 2):
                    # area-based/lanczos don't match with cv2?
                    cv_resized = cv2.resize(cv_img, (new_w, new_h), interpolation=interp)
                    mx_resized = mx.image.resize_short(mx_img, new_size, interp)
                    assert_almost_equal(mx_resized.asnumpy()[:, :, (2, 1, 0)], cv_resized, atol=3)

    def test_color_normalize(self):
        for _ in range(10):
            mean = np.random.rand(3) * 255
            std = np.random.rand(3) + 1
            width = np.random.randint(100, 500)
            height = np.random.randint(100, 500)
            src = np.random.rand(height, width, 3) * 255.
            mx_result = mx.image.color_normalize(mx.nd.array(src),
                mx.nd.array(mean), mx.nd.array(std))
            assert_almost_equal(mx_result.asnumpy(), (src - mean) / std, atol=1e-3)


    def test_imageiter(self):
        im_list = [[np.random.randint(0, 5), x] for x in TestImage.IMAGES]
        test_iter = mx.image.ImageIter(2, (3, 224, 224), label_width=1, imglist=im_list,
            path_root='')
        for _ in range(3):
            for batch in test_iter:
                pass
            test_iter.reset()

        # test with list file
        fname = './data/test_imageiter.lst'
        file_list = ['\t'.join([str(k), str(np.random.randint(0, 5)), x]) \
            for k, x in enumerate(TestImage.IMAGES)]
        with open(fname, 'w') as f:
            for line in file_list:
                f.write(line + '\n')

        test_iter = mx.image.ImageIter(2, (3, 224, 224), label_width=1, path_imglist=fname,
            path_root='')
        for batch in test_iter:
            pass


    def test_augmenters(self):
        # only test if all augmenters will work
        # TODO(Joshua Zhang): verify the augmenter outputs
        im_list = [[0, x] for x in TestImage.IMAGES]
        test_iter = mx.image.ImageIter(2, (3, 224, 224), label_width=1, imglist=im_list,
            resize=640, rand_crop=True, rand_resize=True, rand_mirror=True, mean=True,
            std=np.array([1.1, 1.03, 1.05]), brightness=0.1, contrast=0.1, saturation=0.1,
            hue=0.1, pca_noise=0.1, rand_gray=0.2, inter_method=10, path_root='', shuffle=True)
        for batch in test_iter:
            pass


    def test_image_detiter(self):
        im_list = [_generate_objects() + [x] for x in TestImage.IMAGES]
        det_iter = mx.image.ImageDetIter(2, (3, 300, 300), imglist=im_list, path_root='')
        for _ in range(3):
            for batch in det_iter:
                pass
            det_iter.reset()

        val_iter = mx.image.ImageDetIter(2, (3, 300, 300), imglist=im_list, path_root='')
        det_iter = val_iter.sync_label_shape(det_iter)

        # test file list
        fname = './data/test_imagedetiter.lst'
        im_list = [[k] + _generate_objects() + [x] for k, x in enumerate(TestImage.IMAGES)]
        with open(fname, 'w') as f:
            for line in im_list:
                line = '\t'.join([str(k) for k in line])
                f.write(line + '\n')

        det_iter = mx.image.ImageDetIter(2, (3, 400, 400), path_imglist=fname,
            path_root='')
        for batch in det_iter:
            pass

    def test_det_augmenters(self):
        # only test if all augmenters will work
        # TODO(Joshua Zhang): verify the augmenter outputs
        im_list = [_generate_objects() + [x] for x in TestImage.IMAGES]
        det_iter = mx.image.ImageDetIter(2, (3, 300, 300), imglist=im_list, path_root='',
            resize=640, rand_crop=1, rand_pad=1, rand_gray=0.1, rand_mirror=True, mean=True,
            std=np.array([1.1, 1.03, 1.05]), brightness=0.1, contrast=0.1, saturation=0.1,
            pca_noise=0.1, hue=0.1, inter_method=10, min_object_covered=0.5,
            aspect_ratio_range=(0.2, 5), area_range=(0.1, 4.0), min_eject_coverage=0.5,
            max_attempts=50)
        for batch in det_iter:
            pass

if __name__ == '__main__':
    import nose
    nose.runmodule()
