#!/usr/bin/env python

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

"""Module for ops not belonging to the official numpy package for imperative programming."""

from __future__ import absolute_import
from . import _op
from . import _register
from ._op import *  # pylint: disable=wildcard-import
from ..context import *  # pylint: disable=wildcard-import
# TODO(junwu): revisit what functions should be exposed to users
from ..util import use_np_shape, np_shape, is_np_shape
from ..util import use_np_array, np_array, is_np_array
from ..util import set_np, use_np, reset_np
from ..ndarray import waitall
from .utils import *  # pylint: disable=wildcard-import

__all__ = []
