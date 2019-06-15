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
import warnings
import collections
import ctypes
import mxnet.contrib.amp as amp

def test_amp_coverage():
    conditional = [item[0] for item in amp.lists.symbol.CONDITIONAL_FP32_FUNCS]

    # Check for duplicates
    for a in [amp.lists.symbol.FP16_FUNCS,
          amp.lists.symbol.FP16_FP32_FUNCS,
          amp.lists.symbol.FP32_FUNCS,
          amp.lists.symbol.WIDEST_TYPE_CASTS,
          conditional]:
        ret = [item for item, count in collections.Counter(a).items() if count > 1]
        assert ret == [], "Elements " + str(ret) + " are duplicated in the AMP lists."

    t = []
    for a in [amp.lists.symbol.FP16_FUNCS,
              amp.lists.symbol.FP16_FP32_FUNCS,
              amp.lists.symbol.FP32_FUNCS,
              amp.lists.symbol.WIDEST_TYPE_CASTS,
              conditional]:
        t += a
    ret = [item for item, count in collections.Counter(t).items() if count > 1]
    assert ret == [], "Elements " + str(ret) + " exist in more than 1 AMP list."

    # Check the coverage
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

    ret1 = set(op_names) - set(t)

    if ret1 != set():
        warnings.warn("Operators " + str(ret1) + " do not exist in AMP lists (in "
                       "python/mxnet/contrib/amp/lists/symbol.py) - please add them. "
                       """Please follow these guidelines for choosing a proper list:
                       - if your operator is not to be used in a computational graph
                         (e.g. image manipulation operators, optimizers) or does not have
                         inputs, put it in FP16_FP32_FUNCS list,
                       - if your operator requires FP32 inputs or is not safe to use with lower
                         precision, put it in FP32_FUNCS list,
                       - if your operator supports both FP32 and lower precision, has
                         multiple inputs and expects all inputs to be of the same
                         type, put it in WIDEST_TYPE_CASTS list,
                       - if your operator supports both FP32 and lower precision and has
                         either a single input or supports inputs of different type,
                         put it in FP16_FP32_FUNCS list,
                       - if your operator is both safe to use in lower precision and
                         it is highly beneficial to use it in lower precision, then
                         put it in FP16_FUNCS (this is unlikely for new operators)
                       - If you are not sure which list to choose, FP32_FUNCS is the
                         safest option""")

if __name__ == '__main__':
    test_amp_coverage()
