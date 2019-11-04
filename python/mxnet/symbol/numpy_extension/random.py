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

"""Namespace for operators used in Gluon dispatched by F=symbol."""

from __future__ import absolute_import
from ...context import current_context
from .. import _internal as _npi

__all__ = ['bernoulli']


def bernoulli(probs=None, logits=None, size=None, dtype=None, ctx=None, out=None):
    """
    Sampling from beroulli distributions.
    """
    from ..numpy import _Symbol as np_symbol
    tensor_type_name = np_symbol
    if (probs is None) == (logits is None):
        raise ValueError(
            "Either `probs` or `logits` must be specified, but not both.")
    if dtype is None:
        dtype = 'float32'
    if ctx is None:
        ctx = current_context()
    if size == ():
        size = None
    if probs is not None:
        is_tensor = isinstance(probs, tensor_type_name)
        if is_tensor:
            return _npi.bernoulli(probs, probs=None, logits=None, is_logit=False,
                                  size=size, ctx=ctx, dtype=dtype, out=out)
        else:
            return _npi.bernoulli(probs=probs, logits=None, is_logit=False,
                                  size=size, ctx=ctx, dtype=dtype, out=out)
    else:
        is_tensor = isinstance(logits, tensor_type_name)
        if is_tensor:
            return _npi.bernoulli(logits, probs=None, logits=None, is_logit=True,
                                  size=size, ctx=ctx, dtype=dtype, out=out)
        else:
            return _npi.bernoulli(probs=None, logits=logits, is_logit=True,
                                  size=size, ctx=ctx, dtype=dtype, out=out)
