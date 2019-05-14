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
import collections
import ctypes
import mxnet.amp as amp

def test_amp_coverage():
    for a in [amp.lists.symbol.FP16_FUNCS,
          amp.lists.symbol.FP16_FP32_FUNCS,
          amp.lists.symbol.FP32_FUNCS,
          amp.lists.symbol.WIDEST_TYPE_CASTS]:
        assert([item for item, count in collections.Counter(a).items() if count > 1] == [])
    t = []
    for a in [amp.lists.symbol.FP16_FUNCS,
              amp.lists.symbol.FP16_FP32_FUNCS,
              amp.lists.symbol.FP32_FUNCS,
              amp.lists.symbol.WIDEST_TYPE_CASTS]:
        t += a
    assert([item for item, count in collections.Counter(t).items() if count > 1] == [])
    py_str = lambda x: x.decode('utf-8')

    plist = ctypes.POINTER(ctypes.c_char_p)()
    size = ctypes.c_uint()

    mx.base._LIB.MXListAllOpNames(ctypes.byref(size),
                                     ctypes.byref(plist))
    op_names = []
    for i in range(size.value):
        s = py_str(plist[i])
        if not s.startswith("_backward") \
           and not s.startswith("_contrib_backward_"):
            op_names.append(s)

    assert(op_names.sort() == t.sort())

if __name__ == '__main__':
    test_amp_coverage()
