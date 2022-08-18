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
from __future__ import absolute_import
from distutils.version import StrictVersion
import sys
import pytest
import itertools
import numpy as _np
import platform
import mxnet as mx
import scipy.stats as ss
import scipy.special as scipy_special
from mxnet import np, npx
from mxnet.base import MXNetError
from mxnet.test_utils import assert_almost_equal, use_np, set_default_device
import os
curr_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
sys.path.insert(0, os.path.join(curr_path, '../unittest'))
from common import assertRaises
import random
from mxnet.test_utils import verify_generator, gen_buckets_probs_with_ppf
from mxnet.numpy_op_signature import _get_builtin_op
from mxnet.util import numpy_fallback

set_default_device(mx.gpu(0))

@use_np
@pytest.mark.serial
def test_np_fallback_decorator():
    @numpy_fallback
    def dnp_func(a, b=None, split_inputs=(), ret_type=list):
        """
        Dummy Doc:
        dnp_func is using the same np.xxx operators
        """
        ret_lst = []
        # unsupported indexing case
        ret_lst.append(a[:,a[1,:]>0])
        # unsupported operator
        ret_lst.append(np.nonzero(b))
        # unsupported operator case
        ret_lst.append(tuple(np.split(split_inputs[0], split_inputs[1])))

        return ret_type(ret_lst)

    def onp_func(a, b=None, split_inputs=(), ret_type=list):
        ret_lst = []
        ret_lst.append(a[:,a[1,:]>0])
        ret_lst.append(_np.nonzero(b))
        ret_lst.append(tuple(_np.split(split_inputs[0], split_inputs[1])))
        return ret_type(ret_lst)

    def get_indices(axis_size):
        if axis_size is 0:
            axis_size = random.randint(3, 6)
        samples = random.randint(1, axis_size - 1)
        indices = sorted(random.sample([i for i in range(1, axis_size)], samples))
        indices = tuple(indices)
        return indices

    ret_type = list if random.uniform(0.0, 1.0) > 0.5 else tuple
    mx_a = np.array([[1,2,3],[3,4,5]])
    np_b = _np.random.uniform(size=(3, 4)) > 0.5
    mx_b = np.array(np_b, dtype=np_b.dtype)
    mx_c_len = random.randint(5, 20)
    mx_c = np.random.uniform(size=(mx_c_len,))
    mx_indices = np.array(get_indices(mx_c_len), dtype=np.int64)
    assert dnp_func.__doc__ is not None
    assert 'onp' not in dnp_func.__doc__
    fallback_ret = dnp_func(mx_a, b=mx_b, split_inputs=(mx_c, mx_indices), ret_type=ret_type)
    onp_ret = onp_func(mx_a.asnumpy(), b=mx_b.asnumpy(), split_inputs=(mx_c.asnumpy(), mx_indices.asnumpy()), ret_type=ret_type)
    for fallback_out, onp_out in zip(fallback_ret, onp_ret):
        if isinstance(fallback_out, (list, tuple)):
            for fallback_item, onp_item in zip(fallback_out, onp_out):
                assert fallback_item.device == mx.device.current_device(), f"incorrect output device {str(fallback_item.device)} vs desired {str(mx.device.current_device())}"
                assert isinstance(fallback_item, np.ndarray)
                assert_almost_equal(fallback_item.asnumpy(), onp_item, rtol=1e-3, atol=1e-5, equal_nan=False)
        else:
            assert fallback_out.device == mx.device.current_device(), f"incorrect output device {str(fallback_out.device)} vs desired {str(mx.device.current_device())}"
            assert isinstance(fallback_out, np.ndarray)
            assert_almost_equal(fallback_out.asnumpy(), onp_out, rtol=1e-3, atol=1e-5, equal_nan=False)

    # does not support mixed-device inputs
    assertRaises(AssertionError, dnp_func, mx_a.to_device(npx.cpu(0)), b=mx_b, split_inputs=(mx_c, mx_indices), ret_type=ret_type)
    assertRaises(AssertionError, dnp_func, mx_a, b=mx_b,
                 split_inputs=(mx_c.to_device(npx.cpu(0)), mx_indices.to_device(npx.gpu(0))), ret_type=ret_type)

    @numpy_fallback
    def empty_ret_func():
        return

    # does not support functions with no return values
    assertRaises(ValueError, empty_ret_func)

