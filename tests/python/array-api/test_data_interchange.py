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
from mxnet import np
import torch
import numpy
import pytest


def test_dlpack_torch_mxnet_torch():
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        x = torch.tensor((5,), device='cuda:0', dtype=torch.float64) + 1
    stream.synchronize()
    nx = np.from_dlpack(x)
    assert nx.device == mx.gpu(0)
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        z = torch.from_dlpack(nx)
    stream.synchronize()
    z += 1
    assert z == x

def test_dlpack_mxnet_torch_mxnet():
    x = np.array([5], device=mx.gpu(), dtype="float64") + 1
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        tx = torch.from_dlpack(x)
    stream.synchronize()
    z = np.from_dlpack(tx)
    z += 1
    assert z.device == mx.gpu(0)
    assert z == x

def test_dlpack_error_message():
    with pytest.raises(AttributeError):
        # raise Attribute Error, NumPy array is not PyCapsule or has __dlpack__ attribute
        nx = numpy.array([5])
        x = np.from_dlpack(nx)
    
    with pytest.raises(TypeError):
        # raise TypeError, Stream must be int or None
        stream = torch.cuda.Stream()
        x = np.array([5], device=mx.gpu(), dtype="float64")
        tx = torch.from_dlpack(x.__dlpack__(stream=stream))
    
    with pytest.raises(ValueError):
        # raise ValueError, CPU device has no stream
        x = np.array([5], dtype="float64")
        tx = torch.from_dlpack(x.__dlpack__(stream=0))
