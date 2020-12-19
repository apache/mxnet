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

import mxnet
import numpy as _np
from mxnet import np, npx, _api_internal
from mxnet.ndarray import NDArray
from mxnet.test_utils import use_np

@use_np
def test_str_map():
    amap = mxnet._ffi.convert_to_node({"a": 2, "b": 3})
    assert "a" in amap
    assert len(amap) == 2
    dd = dict(amap.items())
    # TODO add attribute visitor
    # assert amap["a"].value == 2
    assert "a" in dd
    assert "b" in dd

@use_np
def test_string():
    x = mxnet.container.String("xyz")
    assert isinstance(x, mxnet.container.String)
    assert isinstance(x, str)
    assert x.startswith("xy")
    assert x + "1" == "xyz1"
    y = _api_internal._echo(x)
    assert isinstance(y, mxnet.container.String)
    assert x.__mxnet_object__.same_as(y.__mxnet_object__)
    assert x == y

@use_np
def test_string_adt():
    s = mxnet.container.String("xyz")
    arr = mxnet._ffi.convert_to_node([s, s])
    assert arr[0] == s
    assert isinstance(arr[0], mxnet.container.String)

# TODO add attribute visitor
# def test_adt():
#     a = mxnet._ffi.convert_to_node([1, 2, 3])
#     assert len(a) == 3
#     assert a[-1].value == 3
#     a_slice = a[-3:-1]
#     assert (a_slice[0].value, a_slice[1].value) == (1, 2)

# TODO make containers capable of holding NDArrays
# def test_ndarray_container():
#     x = np.array([1, 2, 3])
#     y = np.array([4, 5, 6])
#     arr = mxnet._ffi.convert_to_node([x, y])
#     assert _np.array_equal(arr[0].asnumpy(), x.asnumpy())
#     assert _np.array_equal(arr[1].asnumpy(), y.asnumpy())
#     assert isinstance(arr[0], NDArray)
