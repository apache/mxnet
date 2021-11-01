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
import numpy as np
import torch


def test_dlpack_torch_mxnet_torch():
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        x = torch.tensor((5,), device='cuda:0', dtype=torch.float64) + 1
    mx = np.from_dlpack(x)
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        z = torch.from_dlpack(mx)
    stream.synchronize()
    z += 1
    assert z == x

def test_dlpack_mxnet_torch_mxnet():
    x = np.array([5], ctx=mx.gpu(), dtype="float64") + 1
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        tx = torch.from_dlpack(x)
    stream.synchronize()
    z = np.from_dlpack(tx)
    z += 1
    assert z == x
