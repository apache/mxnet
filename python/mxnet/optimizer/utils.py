# coding: utf-8
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
"""Optimizer utility functions."""
from __future__ import absolute_import


def _flatten_list(nested_list):
    return [item for sublist in nested_list for item in sublist]


def _as_classic(a, allow_np):
    # TODO(junwu): This is a temp solution for allowing converting
    # np.ndarray to mx.nd.NDArray to be fed into the optimizer since
    # users may have custom optimizers implemented using mx.nd.NDArray ops.
    from ..numpy import ndarray as np_ndarray
    if isinstance(a, (tuple, list)):
        if any(isinstance(x, np_ndarray) for x in a):
            if allow_np:
                return [x.as_nd_ndarray() for x in a]
            else:
                raise ValueError('Converting np.ndarray to mx.nd.NDArray is not allowed')
    else:
        if isinstance(a, np_ndarray):
            if allow_np:
                return a.as_nd_ndarray()
            else:
                raise ValueError('Converting np.ndarray to mx.nd.NDArray is not allowed')
    return a
