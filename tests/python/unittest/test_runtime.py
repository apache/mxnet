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
import pytest


def test_features():
    features = Features()
    print(features)
    assert 'CUDA' in features
    assert len(features) >= 30


def test_is_singleton():
    x = Features()
    y = Features()
    assert x is y


def test_is_enabled():
    features = Features()
    for f in features:
        if features[f].enabled:
            assert features.is_enabled(f)
        else:
            assert not features.is_enabled(f)


@pytest.mark.xfail(raises=RuntimeError)
def test_is_enabled_not_existing():
    features = Features()
    features.is_enabled('this girl is on fire')

