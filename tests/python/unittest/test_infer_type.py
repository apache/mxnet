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
import mxnet as mx
import numpy as np
from common import models
from mxnet import autograd
from nose.tools import *

def test_infer_multiout_op():
    data = mx.nd.arange(16, dtype='float64').reshape((4, 4))
    data.attach_grad()

    with autograd.record():
        y = mx.nd.split(data, axis=0, num_outputs=2)
    y[0].backward()
    assert data.grad.dtype == np.float64


if __name__ == "__main__":
    test_infer_multiout_op()