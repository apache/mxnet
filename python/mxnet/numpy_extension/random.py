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

"""Namespace for ops used in imperative programming."""

from __future__ import absolute_import
from .. import random as _mx_rand


__all__ = ['seed']


def seed(seed, ctx='all'):  # pylint: disable=redefined-outer-name
    """Seeds the random number generators in MXNet.

    This affects the behavior of modules in MXNet that uses random number generators,
    like the dropout operator and `ndarray`'s random sampling operators.

    Parameters
    ----------
    seed : int
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
    >>> from mxnet import np, npx
    >>> npx.set_np()
    >>> npx.random.seed(0)
    >>> np.random.uniform()
    array(0.5488135)
    >>> npx.random.seed(128)
    >>> np.random.uniform()
    array(0.03812965)
    >>> npx.random.seed(128)
    >>> np.random.uniform()
    array(0.03812965)
    >>> npx.random.seed(128)
    >>> np.random.uniform(ctx=npx.gpu(0))
    array(0.9894903, ctx=gpu(0))
    >>> npx.random.seed(128)
    >>> np.random.uniform(ctx=npx.gpu(0))
    array(0.9894903, ctx=gpu(0))
    """
    _mx_rand.seed(seed_state=seed, ctx=ctx)
