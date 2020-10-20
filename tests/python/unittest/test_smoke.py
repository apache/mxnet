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

from mxnet import np, npx, use_np, autograd, initializer, gluon
from common import with_environment
import pytest

@use_np
@with_environment('MXNET_ENGINE_TYPE', 'NaiveEngine')
def test_18927():
    """test for no error when dealing with zero-size array in bipartite matching"""
    arr = np.random.rand(0,2)
    arr_grad = np.empty_like(arr)
    autograd.mark_variables([arr], [arr_grad])
    with autograd.record():
        a = npx.bipartite_matching(arr, threshold=0.1)
    a[0].backward()

@use_np
@with_environment('MXNET_ENGINE_TYPE', 'NaiveEngine')
def test_18933_batch_0():
    arr = np.random.rand(0,1,1) # batch = 0
    arr_grad = np.empty_like(arr)
    gamma = np.random.rand(1)
    gamma_grad = np.empty_like(gamma)
    beta = np.random.rand(1)
    beta_grad = np.empty_like(beta)
    autograd.mark_variables([arr, gamma, beta], [arr_grad, gamma_grad, beta_grad])
    with autograd.record():
        a = npx.instance_norm(arr, gamma, beta)
    a.backward()

@use_np
@with_environment('MXNET_ENGINE_TYPE', 'NaiveEngine')
def test_18933_channel_0():
    arr = np.random.rand(1,0,1) # channel = 0
    arr_grad = np.empty_like(arr)
    gamma = np.random.rand(1)
    gamma_grad = np.empty_like(gamma)
    beta = np.random.rand(1)
    beta_grad = np.empty_like(beta)
    autograd.mark_variables([arr, gamma, beta], [arr_grad, gamma_grad, beta_grad])
    with autograd.record():
        a = npx.instance_norm(arr, gamma, beta)
    a.backward()

@use_np
@with_environment('MXNET_ENGINE_TYPE', 'NaiveEngine')
def test_18934_empty_leaky_relu():
    arr = np.random.rand(0,2)
    arr_grad = np.empty_like(arr)

    autograd.mark_variables([arr], [arr_grad])
    with autograd.record():
        res = npx.leaky_relu(arr)
    res.backward()

@use_np
@pytest.mark.parametrize('initializer',[
    'zeros', 'ones', initializer.Constant(3),
    initializer.Uniform(),
    initializer.Normal(),
    initializer.Orthogonal(),
    initializer.Orthogonal(rand_type='normal'),
    initializer.Xavier(),
    initializer.Xavier(rnd_type='gaussian'),
    initializer.MSRAPrelu(),
    initializer.MSRAPrelu(factor_type='in'),
    initializer.MSRAPrelu(factor_type='out'),
    initializer.LSTMBias(),
])
@pytest.mark.parametrize('dtype', [
    'float32', 'float64'
])
def test_19118(initializer, dtype):
    net = gluon.nn.Dense(16, in_units=16)
    net.cast(dtype)
    net.initialize(initializer)
    net.hybridize()
    net(np.zeros((16, 16), dtype=dtype))
