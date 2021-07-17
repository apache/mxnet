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
import pytest
import numpy as onp
import mxnet as mx
import mxnet.ndarray.numpy._internal as _npi
from mxnet import np, npx
from mxnet.test_utils import assert_almost_equal

@pytest.mark.parametrize('a_dtype', [onp.float16, onp.float32, onp.float64])
@pytest.mark.parametrize('b_dtype', [onp.float16, onp.float32, onp.float64])
@pytest.mark.parametrize('shape', [
    (),
    (2, 0, 2, 2),
    (5, 5)
])
@pytest.mark.parametrize('op', [
    '__iadd__', '__iand__', '__ior__', '__ixor__', '__isub__', '__imul__', '__imatmul__',
    '__imod__', '__itruediv__', '__idiv__'])
def test_in_place_dtype(a_dtype, b_dtype, shape, op):
    a = np.random.uniform(size=shape, dtype=a_dtype)
    b = np.random.uniform(size=shape, dtype=b_dtype)
    getattr(a, op)(b)
    assert a.dtype == a_dtype
