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

"""Make builtin ops' signatures compatible with NumPy."""

import inspect
from . import _numpy_op_doc
from . import numpy as mx_np
from . import numpy_extension as mx_npx
from .base import _NP_OP_SUBMODULE_LIST, _NP_EXT_OP_SUBMODULE_LIST, _get_op_submodule_name


def _get_builtin_op(op_name):
    if op_name.startswith('_np_'):
        root_module = mx_np
        op_name_prefix = '_np_'
        submodule_name_list = _NP_OP_SUBMODULE_LIST
    elif op_name.startswith('_npx_'):
        root_module = mx_npx
        op_name_prefix = '_npx_'
        submodule_name_list = _NP_EXT_OP_SUBMODULE_LIST
    else:
        return None

    submodule_name = _get_op_submodule_name(op_name, op_name_prefix, submodule_name_list)
    op_module = root_module
    if len(submodule_name) > 0:
        op_module = getattr(root_module, submodule_name[1:-1], None)
        if op_module is None:
            raise ValueError('Cannot find submodule {} in module {}'
                             .format(submodule_name[1:-1], root_module.__name__))

    op = getattr(op_module, op_name[(len(op_name_prefix)+len(submodule_name)):], None)
    if op is None:
        raise ValueError('Cannot find operator {} in module {}'
                         .format(op_name[len(op_name_prefix):], root_module.__name__))
    return op


def _register_op_signatures():
    for op_name in dir(_numpy_op_doc):
        op = _get_builtin_op(op_name)
        if op is not None:
            op.__signature__ = inspect.signature(getattr(_numpy_op_doc, op_name))


_register_op_signatures()
