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
import sys
from mxnet.runtime import *
from mxnet.base import MXNetError
from nose.tools import *

def test_runtime_features():
    for f in Feature:
        res = has_feature_index(f.value)
        ok_(type(res) is bool)
    for f in features_enabled():
        ok_(type(f) is Feature)
    ok_(type(features_available()) is list)
    ok_(len(features_available()) > 0)
    ok_(len(Feature) > 0)
    print("Features available: {}".format(features_available()))

@raises(MXNetError)
def test_has_feature_2large():
    has_feature_index(sys.maxsize)


if __name__ == "__main__":
    import nose
    nose.runmodule()
