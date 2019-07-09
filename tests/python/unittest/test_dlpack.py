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

import ctypes
import mxnet as mx
from mxnet.base import NDArrayHandle, _LIB, c_str, check_call
from mxnet.test_utils import assert_almost_equal

def test_from_dlpack_backward_compatibility():
    def from_dlpack_old(dlpack):

        PyCapsuleDestructor = ctypes.CFUNCTYPE(None, ctypes.c_void_p)
        _c_str_dltensor = c_str('dltensor')
        _c_str_used_dltensor = c_str('used_dltensor')
        handle = NDArrayHandle()
        dlpack = ctypes.py_object(dlpack)
        assert ctypes.pythonapi.PyCapsule_IsValid(dlpack, _c_str_dltensor), ValueError(
            'Invalid DLPack Tensor. DLTensor capsules can be consumed only once.')
        dlpack_handle = ctypes.c_void_p(ctypes.pythonapi.PyCapsule_GetPointer(dlpack, _c_str_dltensor))
        check_call(_LIB.MXNDArrayFromDLPack(dlpack_handle, ctypes.byref(handle)))
        # Rename PyCapsule (DLPack)
        ctypes.pythonapi.PyCapsule_SetName(dlpack, _c_str_used_dltensor)
        # delete the deleter of the old dlpack
        ctypes.pythonapi.PyCapsule_SetDestructor(dlpack, None)
        return mx.nd.NDArray(handle=handle)

    x = mx.nd.ones((2,3))
    y = mx.nd.to_dlpack_for_read(x)
    z = from_dlpack_old(y)
    assert_almost_equal(x.asnumpy(), z.asnumpy(), rtol=1e-5, atol=1e-5)

if __name__ == '__main__':
    import nose
    nose.runmodule()
