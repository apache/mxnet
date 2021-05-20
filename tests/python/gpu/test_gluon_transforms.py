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
from common import assertRaises
from test_numpy_gluon_data_vision import test_to_tensor, test_normalize, test_crop_resize

set_default_context(mx.gpu(0))

def test_normalize_gpu():
    test_normalize()


def test_to_tensor_gpu():
    test_to_tensor()


@mx.util.use_np
def test_resize_gpu():
    # Test with normal case 3D input float type
    data_in_3d = mx.np.random.uniform(0, 255, (300, 300, 3))
    out_nd_3d = transforms.Resize((100, 100))(data_in_3d)
    data_in_4d_nchw = mx.np.moveaxis(mx.np.expand_dims(data_in_3d, axis=0), 3, 1)
    data_expected_3d = (mx.np.moveaxis(nd.contrib.BilinearResize2D(data_in_4d_nchw.as_nd_ndarray(), height=100, width=100, align_corners=False), 1, 3))[0]
    assert_almost_equal(out_nd_3d.asnumpy(), data_expected_3d.asnumpy())

    # Test with normal case 4D input float type
    data_in_4d = mx.np.random.uniform(0, 255, (2, 300, 300, 3))
    out_nd_4d = transforms.Resize((100, 100))(data_in_4d)
    data_in_4d_nchw = mx.np.moveaxis(data_in_4d, 3, 1)
    data_expected_4d = mx.np.moveaxis(nd.contrib.BilinearResize2D(data_in_4d_nchw.as_nd_ndarray(), height=100, width=100, align_corners=False), 1, 3)
    assert_almost_equal(out_nd_4d.asnumpy(), data_expected_4d.asnumpy())

    # Test invalid interp
    data_in_3d = mx.np.random.uniform(0, 255, (300, 300, 3))
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

def test_crop_resize_gpu():
    test_crop_resize()
