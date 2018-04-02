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
from .context import current_context


def seed(seed_state):
    """Seeds the random number generators in MXNet.

    This affects the behavior of modules in MXNet that uses random number generators,
    like the dropout operator and `NDArray`'s random sampling operators.

    Parameters
    ----------
    seed_state : int
        The random number seed to set to all devices.

    Notes
    -----
    Random number generators in MXNet are device specific. `mx.random.seed(seed_state)` seeds each
    generator with bits which is deterministically generated from `seed_state` and the device id.
    Therefore, random numbers generated from different devices can be different even if they are seeded
    using the same seed. To produce identical random number sequences independent of the device id,
    use `seed_context`.

    Example
    -------
    >>> print(mx.nd.random.normal(shape=(2,2)).asnumpy())
    [[ 1.36481571 -0.62203991]
     [-1.4962182  -0.08511394]]
    >>> print(mx.nd.random.normal(shape=(2,2)).asnumpy())
    [[ 1.09544981 -0.20014545]
     [-0.20808885  0.2527658 ]]
    >>>
    >>> mx.random.seed(128)
    >>> print(mx.nd.random.normal(shape=(2,2)).asnumpy())
    [[ 0.47400656 -0.75213492]
     [ 0.20251541  0.95352972]]
    >>> mx.random.seed(128)
    >>> print(mx.nd.random.normal(shape=(2,2)).asnumpy())
    [[ 0.47400656 -0.75213492]
     [ 0.20251541  0.95352972]]
    """
    if not isinstance(seed_state, int):
        raise ValueError('seed_state must be int')
    seed_state = ctypes.c_int(int(seed_state))
    check_call(_LIB.MXRandomSeed(seed_state))

def seed_context(seed_state, ctx=None):
    """Seeds the random number generator of a device context.

    This affects the behavior of modules in MXNet that uses random number generators,
    like the dropout operator and `NDArray`'s random sampling operators.

    Parameters
    ----------
    seed_state : int
        The random number seed.

    ctx : Context
        The device context of the generator. The default is the current context.

    Notes
    -----
    Seeding with the same number through `mx.random.seed_context` produces the same
    sequence of random numbers independent of the device id, but the sequence can be different
    on different kind of devices as MXNet's random number generators for CPU and GPU use
    different algorithms.

    To seed the random number generators of all devices at once, use `seed`.

    Example
    -------
    # Seeding with `mx.random.seed`. Different results on gpu(0) and gpu(1).
    >>> with mx.Context(mx.gpu(0)):
    ...     mx.random.seed(99)
    ...     print(mx.nd.random.uniform(0, 1, 3))
    [0.29560053 0.07938761 0.29997164]
    <NDArray 3 @gpu(0)>
    >>> with mx.Context(mx.gpu(1)):
    ...     mx.random.seed(99)
    ...     print(mx.nd.random.uniform(0, 1, 3))
    [0.8797334 0.8857584 0.3797555]
    <NDArray 3 @gpu(1)>

    # Seeding with `mx.random.seed_context`. Identical results on gpu(0) and gpu(1).
    # This seeds the generator of the current context. Other generators are not touched.
    # To seed a specific device context, set the optional argument `ctx`.
    >>> with mx.Context(mx.gpu(0)):
    ...     mx.random.seed_context(99)
    ...     print(mx.nd.random.uniform(0, 1, 3))
    [0.29560053 0.07938761 0.29997164]
    <NDArray 3 @gpu(0)>
    >>> with mx.Context(mx.gpu(1)):
    ...     mx.random.seed_context(99)
    ...     print(mx.nd.random.uniform(0, 1, 3))
    [0.29560053 0.07938761 0.29997164]
    <NDArray 3 @gpu(1)>
    """
    if not isinstance(seed_state, int):
        raise ValueError('seed_state must be int')
    if ctx is None:
        ctx = current_context()
    seed_state = ctypes.c_int(int(seed_state))
    check_call(_LIB.MXRandomSeedContext(seed_state, ctx.device_typeid, ctx.device_id))
