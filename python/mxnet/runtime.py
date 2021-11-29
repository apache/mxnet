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

"""Runtime querying of compile time features in the native library.

With this module you can check at runtime which libraries and features were compiled in the library.

Example usage:

.. code-block:: python

    import mxnet
    features=mxnet.runtime.Features()

    features.is_enabled("CUDNN")
    False

    features.is_enabled("CPU_SSE")
    True

    print(features)
    [✖ CUDA, ✖ CUDNN, ✖ NCCL, ✖ TENSORRT, ✔ CPU_SSE, ✔ CPU_SSE2, ✔ CPU_SSE3,
    ✔ CPU_SSE4_1, ✔ CPU_SSE4_2, ✖ CPU_SSE4A, ✔ CPU_AVX, ✖ CPU_AVX2, ✔ OPENMP, ✖ SSE,
    ✔ F16C, ✔ JEMALLOC, ✔ BLAS_OPEN, ✖ BLAS_ATLAS, ✖ BLAS_MKL, ✖ BLAS_APPLE, ✔ LAPACK,
    ✖ ONEDNN, ✔ OPENCV, ✖ DIST_KVSTORE, ✖ INT64_TENSOR_SIZE, ✔ SIGNAL_HANDLER, ✔ DEBUG, ✖ TVM_OP]


"""

import ctypes
import collections
from .base import _LIB, check_call

class Feature(ctypes.Structure):
    """Compile time feature description, member fields: `name` and `enabled`."""
    _fields_ = [
        ("_name", ctypes.c_char_p),
        ("_enabled", ctypes.c_bool)
    ]

    @property
    def name(self):
        """Feature name."""
        return self._name.decode()

    @property
    def enabled(self):
        """True if MXNet was compiled with the given compile-time feature."""
        return self._enabled

    def __repr__(self):
        if self.enabled:
            return "✔ {}".format(self.name)
        else:
            return "✖ {}".format(self.name)

def feature_list():
    """Check the library for compile-time features. The list of features are maintained in libinfo.h and libinfo.cc

    Returns
    -------
    list
        List of :class:`.Feature` objects
    """
    lib_features_c_array = ctypes.POINTER(Feature)()
    lib_features_size = ctypes.c_size_t()
    check_call(_LIB.MXLibInfoFeatures(ctypes.byref(lib_features_c_array), ctypes.byref(lib_features_size)))
    features = [lib_features_c_array[i] for i in range(lib_features_size.value)]
    return features

class Features(collections.OrderedDict):
    """OrderedDict of name to Feature"""
    instance = None
    def __new__(cls):
        if cls.instance is None:
            cls.instance = super(Features, cls).__new__(cls)
            super(Features, cls.instance).__init__([(f.name, f) for f in feature_list()])
        return cls.instance

    def __repr__(self):
        return str(list(self.values()))

    def is_enabled(self, feature_name):
        """Check for a particular feature by name

        Parameters
        ----------
        feature_name: str
            The name of a valid feature as string for example 'CUDA'

        Returns
        -------
        Boolean
            True if it's enabled, False if it's disabled, RuntimeError if the feature is not known
        """
        feature_name = feature_name.upper()
        if feature_name not in self:
            raise RuntimeError("Feature '{}' is unknown, known features are: {}".format(
                feature_name, list(self.keys())))
        return self[feature_name].enabled

def get_branch():
    out = ctypes.c_char_p()
    check_call(_LIB.MXGetBranch(ctypes.byref(out)))
    return out.value.decode('utf-8')

def get_commit_hash():
    out = ctypes.c_char_p()
    check_call(_LIB.MXGetCommitHash(ctypes.byref(out)))
    return out.value.decode('utf-8')
