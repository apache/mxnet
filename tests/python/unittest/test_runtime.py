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

def test_libinfo_features():
    features = libinfo_features()
    print("Lib features: ")
    for f in features:
        print(f.name, f.enabled, f.index)
    ok_(type(features) is list)
    ok_(len(features) > 0)

def test_is_enabled():
    features = libinfo_features()
    for f in features:
        if f.enabled:
            ok_(is_enabled(f.name))
        else:
            ok_(not is_enabled(f.name))

@raises(RuntimeError)
def test_is_enabled_not_existing():
    is_enabled('this girl is on fire')


if __name__ == "__main__":
    import nose
    nose.runmodule()
