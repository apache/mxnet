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
"""Utilities used for translating operators from Onnx to Mxnet."""
# pylint: disable=
from __future__ import absolute_import as _abs

def _fix_attribute_names(attrs, change_map):
    """
    Change attribute names as per values in change_map dictionary.
    Parameters
    ----------
    attrs : dict
        Dict of operator attributes
    change_map : dict
        Dict of onnx attribute name to mxnet attribute names.

    Returns
    -------
    new_attr : dict
        Converted dict of operator attributes.
    """
    new_attr = {}
    for k in attrs.keys():
        if k in change_map:
            new_attr[change_map[k]] = attrs[k]
        else:
            new_attr[k] = attrs[k]
    return new_attr
