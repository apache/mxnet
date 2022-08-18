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
"""Common implementation of Node generic related logic
Acknowledgement: This file originates from incubator-tvm"""
# pylint: disable=unused-import
from numbers import Number, Integral
from .. import _api_internal
from ..ndarray._internal import NDArrayBase
from .object import _ObjectBase, PyNativeObject, _set_node_generic
from .base import string_types

def _scalar_type_inference(value):
    if hasattr(value, 'dtype'):
        dtype = str(value.dtype)
    elif isinstance(value, bool):
        dtype = 'bool'
    elif isinstance(value, float):
        # We intentionally convert the float to float32 since it's more common in DL.
        dtype = 'float32'
    elif isinstance(value, int):
        # We intentionally convert the python int to int32 since it's more common in DL.
        dtype = 'int32'
    else:
        raise NotImplementedError('Cannot automatically inference the type.'
                                  ' value={}'.format(value))
    return dtype


def convert_to_node(value):
    """Convert a python value to corresponding node type.

    Parameters
    ----------
    value : str
        The value to be inspected.

    Returns
    -------
    node : Node
        The corresponding node value.
    """
    if isinstance(value, (_ObjectBase, NDArrayBase, PyNativeObject)):
        return value
    elif isinstance(value, Integral):
        return _api_internal._Integer(value)
    elif isinstance(value, float):
        return _api_internal._Float(value)
    elif isinstance(value, string_types):
        return _api_internal._String(value)
    elif isinstance(value, (list, tuple)):
        value = [convert_to_node(x) for x in value]
        return _api_internal._ADT(*value)
    elif isinstance(value, dict):
        vlist = []
        for item in value.items():
            if (not isinstance(item[0], (_ObjectBase, NDArrayBase, PyNativeObject)) and
                    not isinstance(item[0], string_types)):
                raise ValueError("key of map must already been a container type")
            vlist.append(item[0])
            vlist.append(convert_to_node(item[1]))
        return _api_internal._Map(*vlist)
    raise ValueError(f"don't know how to convert type {type(value)} to node")


def const(value, dtype=None):
    """Construct a constant value for a given type.

    Parameters
    ----------
    value : int or float
        The input value

    dtype : str or None, optional
        The data type.

    Returns
    -------
    expr : Expr
        Constant expression corresponds to the value.
    """
    if dtype is None:
        dtype = _scalar_type_inference(value)
    return _api_internal._const(value, dtype)

_set_node_generic(convert_to_node)
