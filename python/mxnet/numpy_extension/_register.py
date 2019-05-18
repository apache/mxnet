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

"""Registering ops in mxnet.numpy_extension for imperative programming."""

from __future__ import absolute_import

from ..base import _init_np_op_module
from ..ndarray.register import _make_ndarray_function


_init_np_op_module(root_module_name='mxnet', np_module_name='numpy_extension',
                   mx_module_name=None, make_op_func=_make_ndarray_function)
