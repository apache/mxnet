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

# pylint: disable=not-an-iterable

"""runtime querying of compile time features in the native library"""

import ctypes
from .base import _LIB, check_call

class LibFeature(ctypes.Structure):
    """
    Compile time feature description
    """
    _fields_ = [
        ("_name", ctypes.c_char_p),
        ("index", ctypes.c_uint32),
        ("enabled", ctypes.c_bool)
    ]

    @property
    def name(self):
        return self._name.decode()

    def __repr__(self):
        if self.enabled:
            return "✔ {}".format(self.name)
        else:
            return "✖ {}".format(self.name)

def libinfo_features():
    """
    Check the library for compile-time features. The list of features are maintained in libinfo.h and libinfo.cc

    Returns
    -------
    :return: list of class LibFeature indicating which features are available and enabled
    """
    lib_features = ctypes.POINTER(LibFeature)()
    lib_features_size = ctypes.c_size_t()
    check_call(_LIB.MXLibInfoFeatures(ctypes.byref(lib_features), ctypes.byref(lib_features_size)))
    feature_list = [lib_features[i] for i in range(lib_features_size.value)]
    return feature_list

def is_enabled(tocheck):
    """
    Check for a particular feature by name

    Parameters
    ----------
    :param x: str The name of a valid feature as string for example 'CUDA'

    Returns
    -------
    :return: bool True if it's enabled, False if it's disabled, RuntimeError if the feature is not known
    """
    feature_dict = {f.name: f.enabled for f in libinfo_features()}
    if tocheck not in feature_dict:
        raise RuntimeError("Feature '{}' is unknown, known features are: {}".format(tocheck, list(feature_dict.keys())))
    return feature_dict[tocheck]
