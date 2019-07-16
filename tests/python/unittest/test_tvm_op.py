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
from mxnet.test_utils import same
from mxnet.runtime import Features

_features = Features()

def test_tvm_broadcast_add():
    if _features.is_enabled("TVM_OP"):
        a = mx.nd.normal(shape=(2, 3, 4))
        b = mx.nd.normal(shape=(1, 3, 1))
        c = mx.nd.contrib.tvm_vadd(a, b)
        c_np = a.asnumpy() + b.asnumpy()
        assert same(c.asnumpy(), c_np)

if __name__ == '__main__':
    import nose
    nose.runmodule()
