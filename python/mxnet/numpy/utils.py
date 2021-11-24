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

"""Util functions for the numpy module."""



import numpy as onp

__all__ = ['float16', 'float32', 'float64', 'uint8', 'int32', 'int8', 'int64',
           'int16', 'uint16', 'uint32', 'uint64',
           'bool', 'bool_', 'pi', 'inf', 'nan', 'PZERO', 'NZERO', 'newaxis',
           'e', 'NINF', 'PINF', 'NAN', 'NaN',
           '_STR_2_DTYPE_', '_DTYPE_2_STR_', '_type_promotion_table',
           'integer_dtypes', 'floating_dtypes', 'boolean_dtypes', 'numeric_dtypes']

py_bool = bool

float16 = onp.dtype(onp.float16)
float32 = onp.dtype(onp.float32)
float64 = onp.dtype(onp.float64)
uint8 = onp.dtype(onp.uint8)
int32 = onp.dtype(onp.int32)
int8 = onp.dtype(onp.int8)
int64 = onp.dtype(onp.int64)
bool_ = onp.dtype(onp.bool_)
bool = onp.dtype(onp.bool)
int16 = onp.dtype(onp.int16)
uint16 = onp.dtype(onp.uint16)
uint32 = onp.dtype(onp.uint32)
uint64 = onp.dtype(onp.uint64)

pi = onp.pi
inf = onp.inf
nan = onp.nan
PZERO = onp.PZERO
NZERO = onp.NZERO
NINF = onp.NINF
PINF = onp.PINF
e = onp.e
NAN = onp.NAN
NaN = onp.NaN

newaxis = None

_STR_2_DTYPE_ = {'float16': float16, 'float32': float32, 'float64': float64, 'float': float64,
                 'int8': int8, 'int16': int16, 'int32': int32, 'int64': int64, 'int': int64,
                 'uint8': uint8, 'uint16': uint16, 'uint32': uint32, 'uint64': uint64,
                 'bool': bool, 'bool_': bool_, 'None': None}

_DTYPE_2_STR_ = {float16: 'float16', float32: 'float32', float64: 'float64', float: 'float64',
                 int8: 'int8', int16: 'int16', int32: 'int32', int64: 'int64', int:'int64',
                 uint8: 'uint8', uint16: 'uint16', uint32: 'uint32', uint64: 'uint64',
                 bool: 'bool', bool_: 'bool_', py_bool: 'bool', None: 'None'}

_ONP_OP_MODULES = [onp, onp.linalg, onp.random, onp.fft]


def _get_np_op(name):
    """Get official NumPy operator with `name`. If not found, raise ValueError."""
    for mod in _ONP_OP_MODULES:
        op = getattr(mod, name, None)
        if op is not None:
            return op
    raise ValueError('Operator `{}` is not supported by `mxnet.numpy`.'.format(name))


_type_promotion_table = {
    # signed integer type promotion
    (int8, int8): int8,
    (int8, int16): int16,
    (int8, int32): int32,
    (int8, int64): int64,
    (int16, int16): int16,
    (int16, int32): int32,
    (int16, int64): int64,
    (int32, int32): int32,
    (int32, int64): int64,
    (int64, int64): int64,
    # unsigned integer type promotion
    (uint8, uint8): uint8,
    (uint8, uint16): uint16,
    (uint8, uint32): uint32,
    (uint8, uint64): uint64,
    (uint16, uint16): uint16,
    (uint16, uint32): uint32,
    (uint16, uint64): uint64,
    (uint32, uint32): uint32,
    (uint32, uint64): uint64,
    (uint64, uint64): uint64,
    # mixed signed and unsigned integer type promotion
    (int8, uint8): int16,
    (int8, uint16): int32,
    (int8, uint32): int64,
    (int16, uint8): int16,
    (int16, uint16): int32,
    (int16, uint32): int64,
    (int32, uint8): int32,
    (int32, uint16): int32,
    (int32, uint32): int64,
    (int64, uint8): int64,
    (int64, uint16): int64,
    (int64, uint32): int64,
    # float type promotion
    (float16, float16): float16,
    (float16, float32): float32,
    (float16, float64): float64,
    (float32, float32): float32,
    (float32, float64): float64,
    (float64, float64): float64,
    # bool type promotion
    (bool, bool): bool,
    # mixed integer and float16 type promotion
    (int8, float16): float16,
    (int16, float16): float16,
    (int32, float16): float16,
    (int64, float16): float16,
    (uint8, float16): float16,
    (uint16, float16): float16,
    (uint32, float16): float16,
    (uint64, float16): float16,
    # mixed integer and float16 type promotion
    (int8, float32): float32,
    (int16, float32): float32,
    (int32, float32): float32,
    (int64, float32): float32,
    (uint8, float32): float32,
    (uint16, float32): float32,
    (uint32, float32): float32,
    (uint64, float32): float32,
    # mixed integer and float32 type promotion
    (int8, float32): float32,
    (int16, float32): float32,
    (int32, float32): float32,
    (int64, float32): float32,
    (uint8, float32): float32,
    (uint16, float32): float32,
    (uint32, float32): float32,
    (uint64, float32): float32,
    # mixed integer and float64 type promotion
    (int8, float64): float64,
    (int16, float64): float64,
    (int32, float64): float64,
    (int64, float64): float64,
    (uint8, float64): float64,
    (uint16, float64): float64,
    (uint32, float64): float64,
    (uint64, float64): float64,
    # mixed bool and other type promotion
    (bool, int8): int8,
    (bool, int16): int16,
    (bool, int32): int32,
    (bool, int64): int64,
    (bool, uint8): uint8,
    (bool, uint16): uint16,
    (bool, uint32): uint32,
    (bool, uint64): uint64,
    (bool, float16): float16,
    (bool, float32): float32,
    (bool, float64): float64,
}

integer_dtypes = [
    int8,
    int16,
    int32,
    int64,
    uint8,
    uint16,
    uint32,
    uint64,
]

floating_dtypes = [
    float16,
    float32,
    float64,
]

numeric_dtypes = [
    *integer_dtypes,
    *floating_dtypes,
]

boolean_dtypes = [
    bool_,
]
