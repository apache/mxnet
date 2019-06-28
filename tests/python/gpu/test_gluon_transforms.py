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
import os
import sys
import mxnet as mx
import mxnet.ndarray as nd
import numpy as np
from mxnet import gluon
from mxnet.base import MXNetError
from mxnet.gluon.data.vision import transforms
from mxnet.test_utils import assert_almost_equal, set_default_context
from mxnet.test_utils import almost_equal, same
curr_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
sys.path.insert(0, os.path.join(curr_path, '../unittest'))
from common import assertRaises, setup_module, with_seed, teardown


set_default_context(mx.gpu(0))

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

    # Default normalize values i.e., mean=0, std=1
    data_in_3d_def = nd.random.uniform(0, 1, (3, 300, 300))
    out_nd_3d_def = transforms.Normalize()(data_in_3d_def)
    data_expected_3d_def = data_in_3d_def.asnumpy()
    assert_almost_equal(data_expected_3d_def, out_nd_3d_def.asnumpy())

    # Invalid Input - Neither 3D or 4D input
    invalid_data_in = nd.random.uniform(0, 1, (5, 5, 3, 300, 300))
    normalize_transformer = transforms.Normalize(mean=(0, 1, 2), std=(3, 2, 1))
    assertRaises(MXNetError, normalize_transformer, invalid_data_in)

    # Invalid Input - Channel neither 1 or 3
    invalid_data_in = nd.random.uniform(0, 1, (5, 4, 300, 300))
    normalize_transformer = transforms.Normalize(mean=(0, 1, 2), std=(3, 2, 1))
    assertRaises(MXNetError, normalize_transformer, invalid_data_in)

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
def test_resize():
    # Test with normal case 3D input float type
    data_in_3d = nd.random.uniform(0, 255, (300, 300, 3))
    out_nd_3d = transforms.Resize((100, 100))(data_in_3d)
    data_in_4d_nchw = nd.moveaxis(nd.expand_dims(data_in_3d, axis=0), 3, 1)
    data_expected_3d = (nd.moveaxis(nd.contrib.BilinearResize2D(data_in_4d_nchw, height=100, width=100), 1, 3))[0]
    assert_almost_equal(out_nd_3d.asnumpy(), data_expected_3d.asnumpy())

    # Test with normal case 4D input float type
    data_in_4d = nd.random.uniform(0, 255, (2, 300, 300, 3))
    out_nd_4d = transforms.Resize((100, 100))(data_in_4d)
    data_in_4d_nchw = nd.moveaxis(data_in_4d, 3, 1)
    data_expected_4d = nd.moveaxis(nd.contrib.BilinearResize2D(data_in_4d_nchw, height=100, width=100), 1, 3)
    assert_almost_equal(out_nd_4d.asnumpy(), data_expected_4d.asnumpy())

    # Test invalid interp
    data_in_3d = nd.random.uniform(0, 255, (300, 300, 3))
    invalid_transform = transforms.Resize(-150, keep_ratio=False, interpolation=2)
    assertRaises(MXNetError, invalid_transform, data_in_3d)

    # Credited to Hang Zhang
    def py_bilinear_resize_nhwc(x, outputHeight, outputWidth):
        batch, inputHeight, inputWidth, channel = x.shape
        if outputHeight == inputHeight and outputWidth == inputWidth:
            return x
        y = np.empty([batch, outputHeight, outputWidth, channel]).astype('uint8')
        rheight = 1.0 * (inputHeight - 1) / (outputHeight - 1) if outputHeight > 1 else 0.0
        rwidth = 1.0 * (inputWidth - 1) / (outputWidth - 1) if outputWidth > 1 else 0.0
        for h2 in range(outputHeight):
            h1r = 1.0 * h2 * rheight
            h1 = int(np.floor(h1r))
            h1lambda = h1r - h1
            h1p = 1 if h1 < (inputHeight - 1) else 0
            for w2 in range(outputWidth):
                w1r = 1.0 * w2 * rwidth
                w1 = int(np.floor(w1r))
                w1lambda = w1r - w1
                w1p = 1 if w1 < (inputHeight - 1) else 0
                for b in range(batch):
                    for c in range(channel):
                        y[b][h2][w2][c] = (1-h1lambda)*((1-w1lambda)*x[b][h1][w1][c] + \
                            w1lambda*x[b][h1][w1+w1p][c]) + \
                            h1lambda*((1-w1lambda)*x[b][h1+h1p][w1][c] + \
                            w1lambda*x[b][h1+h1p][w1+w1p][c])
        return y

    # Test with normal case 3D input int8 type
    data_in_4d = nd.random.uniform(0, 255, (1, 300, 300, 3)).astype('uint8')
    out_nd_3d = transforms.Resize((100, 100))(data_in_4d[0])
    assert_almost_equal(out_nd_3d.asnumpy(), py_bilinear_resize_nhwc(data_in_4d.asnumpy(), 100, 100)[0], atol=1.0)

    # Test with normal case 4D input int8 type
    data_in_4d = nd.random.uniform(0, 255, (2, 300, 300, 3)).astype('uint8')
    out_nd_4d = transforms.Resize((100, 100))(data_in_4d)
    assert_almost_equal(out_nd_4d.asnumpy(), py_bilinear_resize_nhwc(data_in_4d.asnumpy(), 100, 100), atol=1.0)
