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
# pylint: disable=not-an-iterable

"""runtime detection of compile time features in the native library"""

import ctypes
import enum
from .base import _LIB, check_call, mx_uint, py_str


def _feature_names_available():
    """

    :return:
    """
    feature_list = ctypes.POINTER(ctypes.c_char_p)()
    feature_list_sz = ctypes.c_size_t()
    check_call(_LIB.MXRuntimeFeatureList(ctypes.byref(feature_list_sz), ctypes.byref(feature_list)))
    feature_names = []
    for i in range(feature_list_sz.value):
        feature_names.append(py_str(feature_list[i]))
    return feature_names

Feature = enum.Enum('Feature', {name: index for index, name in enumerate(_feature_names_available())})

def features_available():
    """
    Returns
    -------
    features: list of Feature enum
        Features available in the backend which includes disabled and enabled ones
    """
    return list(Feature)

def has_feature_index(feature):
    """
    Check the library for compile-time feature at runtime

    Parameters
    ----------
    feature : int
        An integer representing the feature to check

    Returns
    -------
    boolean
        True if the feature is enabled, false otherwise
    """
    res = ctypes.c_bool()
    check_call(_LIB.MXRuntimeHasFeature(mx_uint(feature), ctypes.byref(res)))
    return res.value


def features_enabled():
    """
    Returns
    -------
    features: list of Feature enum
        list of enabled features in the back-end
    """
    res = []
    for f in Feature:
        if has_feature_index(f.value):
            res.append(f)
    return res
