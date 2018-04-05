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

# coding: utf-8
# pylint: disable=no-member, protected-access, unused-import, no-name-in-module
# pylint: disable=wildcard-import, unused-wildcard-import
"""Random number interface of MXNet."""
from __future__ import absolute_import

import ctypes
from .base import _LIB, check_call
from .ndarray.random import *
from .context import Context


def seed(seed_state, ctx="all"):
    """Seeds the random number generators in MXNet.

    This affects the behavior of modules in MXNet that uses random number generators,
    like the dropout operator and `NDArray`'s random sampling operators.

    Parameters
    ----------
    seed_state : int
        The random number seed.

    ctx : Context
        The device context of the generator. The default is "all" which means seeding random
        number generators of all devices.

    Notes
    -----
    Random number generators in MXNet are device specific.
    `mx.random.seed(seed_state)` sets the state of each generator using `seed_state` and the
    device id. Therefore, random numbers generated from different devices can be different
    even if they are seeded using the same seed.

    To produce identical random number sequences independent of the device id,
    set optional `ctx` argument. This produces the same sequence of random numbers independent
    of the device id, but the sequence can be different on different kind of devices as MXNet's
    random number generators for CPU and GPU use different algorithms.

    Example
    -------
    >>> print(mx.nd.random.normal(shape=(2,2)).asnumpy())
    [[ 1.36481571 -0.62203991]
     [-1.4962182  -0.08511394]]
    >>> print(mx.nd.random.normal(shape=(2,2)).asnumpy())
    [[ 1.09544981 -0.20014545]
     [-0.20808885  0.2527658 ]]
    # Same results on the same device with the same seed
    >>> mx.random.seed(128)
    >>> print(mx.nd.random.normal(shape=(2,2)).asnumpy())
    [[ 0.47400656 -0.75213492]
     [ 0.20251541  0.95352972]]
    >>> mx.random.seed(128)
    >>> print(mx.nd.random.normal(shape=(2,2)).asnumpy())
    [[ 0.47400656 -0.75213492]
     [ 0.20251541  0.95352972]]
    # Different results on gpu(0) and gpu(1) with the same seed
    >>> mx.random.seed(128)
    >>> print(mx.nd.random.normal(shape=(2,2), ctx=mx.gpu(0)).asnumpy())
    [[ 2.5020072 -1.6884501]
     [-0.7931333 -1.4218881]]
    >>> mx.random.seed(128)
    >>> print(mx.nd.random.normal(shape=(2,2), ctx=mx.gpu(1)).asnumpy())
    [[ 0.24336822 -1.664805  ]
     [-1.0223296   1.253198  ]]
    # Seeding with `ctx` argument produces identical results on gpu(0) and gpu(1)
    >>> mx.random.seed(128, ctx=mx.gpu(0))
    >>> print(mx.nd.random.normal(shape=(2,2), ctx=mx.gpu(0)).asnumpy())
    [[ 2.5020072 -1.6884501]
     [-0.7931333 -1.4218881]]
    >>> mx.random.seed(128, ctx=mx.gpu(1))
    >>> print(mx.nd.random.normal(shape=(2,2), ctx=mx.gpu(1)).asnumpy())
    [[ 2.5020072 -1.6884501]
     [-0.7931333 -1.4218881]]
    """
    if not isinstance(seed_state, int):
        raise ValueError('seed_state must be int')
    seed_state = ctypes.c_int(seed_state)
    if ctx == "all":
        check_call(_LIB.MXRandomSeed(seed_state))
    else:
        ctx = Context(ctx)
        check_call(_LIB.MXRandomSeedContext(seed_state, ctx.device_typeid, ctx.device_id))
