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
from mxnet.test_utils import almost_equal
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
def test_resize():
    # Test with normal case 3D input
    data_in_3d = nd.random.uniform(0, 255, (300, 300, 3)).astype('uint8')
    out_nd_3d = transforms.Resize((100, 100))(data_in_3d)
    data_in_4d_nchw = nd.moveaxis(nd.expand_dims(data_in_3d, axis=0), 3, 1)
    data_expected_3d = (nd.moveaxis(nd.contrib.BilinearResize2D(data_in_4d_nchw, 100, 100), 1, 3))[0]
    assert_almost_equal(out_nd_3d.asnumpy(), data_expected_3d.asnumpy())
