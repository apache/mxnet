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
from .base import _LIB, check_call, mx_uint

feature_names = [
    "CUDA",
    "CUDNN",
    "NCCL",
    "CUDA_RTC",
    "TENSORRT",
    "CPU_SSE",
    "CPU_SSE2",
    "CPU_SSE3",
    "CPU_SSE4_1",
    "CPU_SSE4_2",
    "CPU_SSE4A",
    "CPU_AVX",
    "CPU_AVX2",
    "OPENMP",
    "SSE",
    "F16C",
    "JEMALLOC",
    "BLAS_OPEN",
    "BLAS_ATLAS",
    "BLAS_MKL",
    "BLAS_APPLE",
    "LAPACK",
    "MKLDNN",
    "OPENCV",
    "CAFFE",
    "PROFILER",
    "DIST_KVSTORE",
    "CXX14",
    "SIGNAL_HANDLER",
    "DEBUG"
]


Feature = enum.Enum('Feature', {name: index for index, name in enumerate(feature_names)})


def has_feature(feature):
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
    check_call(_LIB.MXHasFeature(mx_uint(feature), ctypes.byref(res)))
    return res.value


def features_enabled():
    """
    Returns
    -------
    features: list of Feature
        list of enabled features in the back-end
    """
    res = []
    for f in Feature:
        if has_feature(f.value):
            res.append(f)
    return res

def features_enabled_str(sep=', '):
    """
    Returns
    -------
    string with a comma separated list of enabled features in the back-end. For example:
    "CPU_SSE, OPENMP, F16C, LAPACK, MKLDNN, OPENCV, SIGNAL_HANDLER, DEBUG"
    """
    return sep.join(map(lambda x: x.name, features_enabled()))
