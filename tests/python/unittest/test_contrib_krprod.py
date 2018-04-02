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

from __future__ import print_function
import numpy as np
import mxnet as mx

from numpy.testing import assert_allclose

def assert_mx_allclose(A, B, **kwds):
    return assert_allclose(A.asnumpy(), B.asnumpy(), **kwds)


def test_krprod_one_input():
    A = mx.nd.arange(1,9).reshape((2,4))
    out = mx.nd.khatri_rao(A)
    assert_mx_allclose(out, A, rtol=1e-12)


def test_krprod_two_inputs():
    A = mx.nd.arange(1,7).reshape((3,2))
    B = mx.nd.arange(1,3).reshape((1,2))
    out = mx.nd.khatri_rao(A, B)
    expected = mx.nd.array([[1,4],[3,8],[5,12]])
    assert_mx_allclose(out, expected, rtol=1e-12)

    A = mx.nd.arange(1,7).reshape((3,2))
    B = mx.nd.arange(1,9).reshape((4,2))
    out = mx.nd.khatri_rao(A, B)
    expected = mx.nd.array([[1,4],[3,8],[5,12],[7,16],[3,8],[9,16],[15,24],
                            [21,32],[5,12],[15,24],[25,36],[35,48]])
    assert_mx_allclose(out, expected, rtol=1e-12)


def test_krprod_three_inputs():
    A = mx.nd.arange(1,7).reshape((3,2))
    B = mx.nd.arange(1,3).reshape((1,2))
    C = mx.nd.arange(1,5).reshape((2,2))
    out = mx.nd.khatri_rao(A, B, C)
    expected = mx.nd.array([[1,8],[3,16],[3,16],[9,32],[5,24],[15,48]])
    assert_mx_allclose(out, expected, rtol=1e-12)

    out_AB = mx.nd.khatri_rao(A, B)
    out = mx.nd.khatri_rao(out_AB, C)
    assert_mx_allclose(out, expected, rtol=1e-12)

    out_BC = mx.nd.khatri_rao(B, C)
    out = mx.nd.khatri_rao(A, out_BC)
    assert_mx_allclose(out, expected, rtol=1e-12)
