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
from collections import namedtuple

import mxnet as mx
import mxnet.ndarray as nd
from mxnet.base import MXNetError
from mxnet import gluon
from mxnet.base import MXNetError
from mxnet.gluon.data.vision import transforms
from mxnet import image
from mxnet.test_utils import *
from common import assertRaises, setup_module, with_seed, teardown

import numpy as np

@with_seed()
def test_to_tensor():
    # 3D Input
    data_in = np.random.uniform(0, 255, (300, 300, 3)).astype(dtype=np.uint8)
    out_nd = transforms.ToTensor()(nd.array(data_in, dtype='uint8'))
    assert_almost_equal(out_nd.asnumpy(), np.transpose(
                        data_in.astype(dtype=np.float32) / 255.0, (2, 0, 1)))

    # 4D Input
    data_in = np.random.uniform(0, 255, (5, 300, 300, 3)).astype(dtype=np.uint8)
    out_nd = transforms.ToTensor()(nd.array(data_in, dtype='uint8'))
    assert_almost_equal(out_nd.asnumpy(), np.transpose(
                        data_in.astype(dtype=np.float32) / 255.0, (0, 3, 1, 2)))
    
    # Invalid Input
    invalid_data_in = nd.random.uniform(0, 255, (5, 5, 300, 300, 3)).astype(dtype=np.uint8)
    transformer = transforms.ToTensor()
    assertRaises(MXNetError, transformer, invalid_data_in)
    
    # Bounds (0->0, 255->1)
    data_in = np.zeros((10, 20, 3)).astype(dtype=np.uint8)
    out_nd = transforms.ToTensor()(nd.array(data_in, dtype='uint8'))
    assert same(out_nd.asnumpy(), np.transpose(np.zeros(data_in.shape, dtype=np.float32), (2, 0, 1)))

    data_in = np.full((10, 20, 3), 255).astype(dtype=np.uint8)
    out_nd = transforms.ToTensor()(nd.array(data_in, dtype='uint8'))
    assert same(out_nd.asnumpy(), np.transpose(np.ones(data_in.shape, dtype=np.float32), (2, 0, 1)))


@with_seed()
def test_normalize():
    # 3D Input
    data_in_3d = nd.random.uniform(0, 1, (3, 300, 300))
    out_nd_3d = transforms.Normalize(mean=(0, 1, 2), std=(3, 2, 1))(data_in_3d)
    data_expected_3d = data_in_3d.asnumpy()
    data_expected_3d[:][:][0] = data_expected_3d[:][:][0] / 3.0
    data_expected_3d[:][:][1] = (data_expected_3d[:][:][1] - 1.0) / 2.0
    data_expected_3d[:][:][2] = data_expected_3d[:][:][2] - 2.0
    assert_almost_equal(data_expected_3d, out_nd_3d.asnumpy())

    # 4D Input
    data_in_4d = nd.random.uniform(0, 1, (2, 3, 300, 300))
    out_nd_4d = transforms.Normalize(mean=(0, 1, 2), std=(3, 2, 1))(data_in_4d)
    data_expected_4d = data_in_4d.asnumpy()
    data_expected_4d[0][:][:][0] = data_expected_4d[0][:][:][0] / 3.0
    data_expected_4d[0][:][:][1] = (data_expected_4d[0][:][:][1] - 1.0) / 2.0
    data_expected_4d[0][:][:][2] = data_expected_4d[0][:][:][2] - 2.0
    data_expected_4d[1][:][:][0] = data_expected_4d[1][:][:][0] / 3.0
    data_expected_4d[1][:][:][1] = (data_expected_4d[1][:][:][1] - 1.0) / 2.0
    data_expected_4d[1][:][:][2] = data_expected_4d[1][:][:][2] - 2.0
    assert_almost_equal(data_expected_4d, out_nd_4d.asnumpy())

    # Invalid Input - Neither 3D or 4D input
    invalid_data_in = nd.random.uniform(0, 1, (5, 5, 3, 300, 300))
    normalize_transformer = transforms.Normalize(mean=(0, 1, 2), std=(3, 2, 1))
    assertRaises(MXNetError, normalize_transformer, invalid_data_in)

    # Invalid Input - Channel neither 1 or 3
    invalid_data_in = nd.random.uniform(0, 1, (5, 4, 300, 300))
    normalize_transformer = transforms.Normalize(mean=(0, 1, 2), std=(3, 2, 1))
    assertRaises(MXNetError, normalize_transformer, invalid_data_in)


@with_seed()
def test_resize():
    def _test_resize_with_diff_type(dtype):
        # test normal case
        data_in = nd.random.uniform(0, 255, (300, 200, 3)).astype(dtype)
        out_nd = transforms.Resize(200)(data_in)
        data_expected = mx.image.imresize(data_in, 200, 200, 1)
        assert_almost_equal(out_nd.asnumpy(), data_expected.asnumpy())
        # test 4D input
        data_bath_in = nd.random.uniform(0, 255, (3, 300, 200, 3)).astype(dtype)
        out_batch_nd = transforms.Resize(200)(data_bath_in)
        for i in range(len(out_batch_nd)):
            assert_almost_equal(mx.image.imresize(data_bath_in[i], 200, 200, 1).asnumpy(),
                out_batch_nd[i].asnumpy())
        # test interp = 2
        out_nd = transforms.Resize(200, interpolation=2)(data_in)
        data_expected = mx.image.imresize(data_in, 200, 200, 2)
        assert_almost_equal(out_nd.asnumpy(), data_expected.asnumpy())
        # test height not equals to width
        out_nd = transforms.Resize((200, 100))(data_in)
        data_expected = mx.image.imresize(data_in, 200, 100, 1)
        assert_almost_equal(out_nd.asnumpy(), data_expected.asnumpy())
        # test keep_ratio
        out_nd = transforms.Resize(150, keep_ratio=True)(data_in)
        data_expected = mx.image.imresize(data_in, 150, 225, 1)
        assert_almost_equal(out_nd.asnumpy(), data_expected.asnumpy())
        # test size below zero
        invalid_transform = transforms.Resize(-150, keep_ratio=True)
        assertRaises(MXNetError, invalid_transform, data_in)
        # test size more than 2:
        invalid_transform = transforms.Resize((100, 100, 100), keep_ratio=True)
        assertRaises(MXNetError, invalid_transform, data_in)

    for dtype in ['uint8', 'float32', 'float64']:
        _test_resize_with_diff_type(dtype)    


@with_seed()
def test_crop_resize():
    def _test_crop_resize_with_diff_type(dtype):
        # test normal case
        data_in = nd.arange(60).reshape((5, 4, 3)).astype(dtype)
        out_nd = transforms.CropResize(0, 0, 3, 2)(data_in)
        out_np = out_nd.asnumpy()
        assert(out_np.sum() == 180)
        assert((out_np[0:2,1,1].flatten() == [4, 16]).all())
        # test 4D input
        data_bath_in = nd.arange(180).reshape((2, 6, 5, 3)).astype(dtype)
        out_batch_nd = transforms.CropResize(1, 2, 3, 4)(data_bath_in)
        out_batch_np = out_batch_nd.asnumpy()
        assert(out_batch_np.sum() == 7524)
        assert((out_batch_np[0:2,0:4,1,1].flatten() == [37,  52,  67,  82, 127, 142, 157, 172]).all())
        # test normal case with resize
        data_in = nd.random.uniform(0, 255, (300, 200, 3)).astype(dtype)
        out_nd = transforms.CropResize(0, 0, 100, 50, (25, 25), 2)(data_in)
        data_expected = image.imresize(nd.slice(data_in, (0, 0, 0), (50, 100 , 3)), 25, 25, 2)
        assert_almost_equal(out_nd.asnumpy(), data_expected.asnumpy())
        # test 4D input with resize
        data_bath_in = nd.random.uniform(0, 255, (3, 300, 200, 3)).astype(dtype)
        out_batch_nd = transforms.CropResize(0, 0, 100, 50, (25, 25), 2)(data_bath_in)
        for i in range(len(out_batch_nd)):
            assert_almost_equal(image.imresize(nd.slice(data_bath_in[i], (0, 0, 0), (50, 100, 3)), 25, 25, 2).asnumpy(),
                out_batch_nd[i].asnumpy())
        # test with resize height and width should be greater than 0
        transformer = transforms.CropResize(0, 0, 100, 50, (-25, 25), 2)
        assertRaises(MXNetError, transformer, data_in)
        # test height and width should be greater than 0 
        transformer = transforms.CropResize(0, 0, -100, -50)
        assertRaises(MXNetError, transformer, data_in)
        # test cropped area is bigger than input data
        transformer = transforms.CropResize(150, 200, 200, 500)
        assertRaises(MXNetError, transformer, data_in)
        assertRaises(MXNetError, transformer, data_bath_in)

    for dtype in ['uint8', 'float32', 'float64']:
        _test_crop_resize_with_diff_type(dtype)  

    # test nd.image.crop backward
    def test_crop_backward(test_nd_arr, TestCase):
        a_np = test_nd_arr.asnumpy()
        b_np = a_np[(slice(TestCase.y, TestCase.y + TestCase.height), slice(TestCase.x, TestCase.x + TestCase.width), slice(0, 3))]

        data = mx.sym.Variable('data')
        crop_sym = mx.sym.image.crop(data, TestCase.x, TestCase.y, TestCase.width, TestCase.height)

        expected_in_grad = np.zeros_like(a_np)
        expected_in_grad[(slice(TestCase.y, TestCase.y + TestCase.height), slice(TestCase.x, TestCase.x + TestCase.width), slice(0, 3))] = b_np
        check_symbolic_backward(crop_sym, [a_np], [b_np], [expected_in_grad])

    TestCase = namedtuple('TestCase', ['x', 'y', 'width', 'height'])
    test_list = [TestCase(0, 0, 3, 3), TestCase(2, 1, 1, 2), TestCase(0, 1, 3, 2)]

    for dtype in ['uint8', 'float32', 'float64']:
        data_in = nd.arange(60).reshape((5, 4, 3)).astype(dtype)
        for test_case in test_list:
            test_crop_backward(data_in, test_case)
        


    # check numeric gradient of nd.image.crop
    # in_data = np.arange(36).reshape(3, 4, 3)
    # data = mx.sym.Variable('data')
    # image_crop_sym = mx.sym.image.crop(data, 0, 0, 2, 2)
    # check_numeric_gradient(image_crop_sym, [in_data])


@with_seed()
def test_flip_left_right():
    data_in = np.random.uniform(0, 255, (300, 300, 3)).astype(dtype=np.uint8)
    flip_in = data_in[:, ::-1, :]
    data_trans = nd.image.flip_left_right(nd.array(data_in, dtype='uint8'))
    assert_almost_equal(flip_in, data_trans.asnumpy())


@with_seed()
def test_flip_top_bottom():
    data_in = np.random.uniform(0, 255, (300, 300, 3)).astype(dtype=np.uint8)
    flip_in = data_in[::-1, :, :]
    data_trans = nd.image.flip_top_bottom(nd.array(data_in, dtype='uint8'))
    assert_almost_equal(flip_in, data_trans.asnumpy())


@with_seed()
def test_transformer():
    from mxnet.gluon.data.vision import transforms

    transform = transforms.Compose([
        transforms.Resize(300),
        transforms.Resize(300, keep_ratio=True),
        transforms.CenterCrop(256),
        transforms.RandomResizedCrop(224),
        transforms.RandomFlipLeftRight(),
        transforms.RandomColorJitter(0.1, 0.1, 0.1, 0.1),
        transforms.RandomBrightness(0.1),
        transforms.RandomContrast(0.1),
        transforms.RandomSaturation(0.1),
        transforms.RandomHue(0.1),
        transforms.RandomLighting(0.1),
        transforms.ToTensor(),
        transforms.Normalize([0, 0, 0], [1, 1, 1])])

    transform(mx.nd.ones((245, 480, 3), dtype='uint8')).wait_to_read()



if __name__ == '__main__':
    import nose
    nose.runmodule()
