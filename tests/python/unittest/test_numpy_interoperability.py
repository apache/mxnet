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

# pylint: skip-file
from __future__ import absolute_import
from __future__ import division
from distutils.version import StrictVersion
import sys
import platform
import itertools
import numpy as _np
import unittest
import pytest
from mxnet import np, util
from mxnet.test_utils import assert_almost_equal
from mxnet.test_utils import use_np
from mxnet.test_utils import is_op_runnable
from common import assertRaises, random_seed
from mxnet.numpy_dispatch_protocol import with_array_function_protocol, with_array_ufunc_protocol
from mxnet.numpy_dispatch_protocol import _NUMPY_ARRAY_FUNCTION_LIST, _NUMPY_ARRAY_UFUNC_LIST


_INT_DTYPES = [np.int8, np.int32, np.int64, np.uint8]
_FLOAT_DTYPES = [np.float16, np.float32, np.float64]
_DTYPES = _INT_DTYPES + _FLOAT_DTYPES
_TVM_OPS = [
    'equal',
    'not_equal',
    'less',
    'less_equal',
    'greater',
    'greater_equal',
    'logical_and',
    'logical_or',
    'logical_xor',
]


class OpArgMngr(object):
    """Operator argument manager for storing operator workloads."""
    _args = {}

    @staticmethod
    def add_workload(name, *args, **kwargs):
        if name not in OpArgMngr._args:
            OpArgMngr._args[name] = []
        OpArgMngr._args[name].append({'args': args, 'kwargs': kwargs})

    @staticmethod
    def get_workloads(name):
        if OpArgMngr._args == {}:
            _prepare_workloads()
        return OpArgMngr._args.get(name, None)

    @staticmethod
    def randomize_workloads():
        # Force a new _prepare_workloads(), which will be based on new random numbers
        OpArgMngr._args = {}


def _add_workload_all():
    # check bad element in all positions
    for i in range(256-7):
        e = np.array([True] * 256, dtype=bool)[7::]
        e[i] = False
        OpArgMngr.add_workload('all', e)
    # big array test for blocked libc loops
    for i in list(range(9, 6000, 507)) + [7764, 90021, -10]:
        e = np.array([True] * 100043, dtype=bool)
        e[i] = False
        OpArgMngr.add_workload('all', e)


def _add_workload_any():
    # check bad element in all positions
    for i in range(256-7):
        d = np.array([False] * 256, dtype=bool)[7::]
        d[i] = True
        OpArgMngr.add_workload('any', d)
    # big array test for blocked libc loops
    for i in list(range(9, 6000, 507)) + [7764, 90021, -10]:
        d = np.array([False] * 100043, dtype=bool)
        d[i] = True
        OpArgMngr.add_workload('any', d)


def _add_workload_sometrue():
    # check bad element in all positions
    for i in range(256-7):
        d = np.array([False] * 256, dtype=bool)[7::]
        d[i] = True
        OpArgMngr.add_workload('sometrue', d)
    # big array test for blocked libc loops
    for i in list(range(9, 6000, 507)) + [7764, 90021, -10]:
        d = np.array([False] * 100043, dtype=bool)
        d[i] = True
        OpArgMngr.add_workload('sometrue', d)


def _add_workload_unravel_index():
    OpArgMngr.add_workload('unravel_index', indices=np.array([2],dtype=_np.int64), shape=(2, 2))
    OpArgMngr.add_workload('unravel_index', np.array([(2*3 + 1)*6 + 4], dtype=_np.int64), (4, 3, 6))
    OpArgMngr.add_workload('unravel_index', np.array([22, 41, 37], dtype=_np.int32), (7, 6))
    OpArgMngr.add_workload('unravel_index', np.array([1621],dtype=_np.uint8), (6, 7, 8, 9))
    OpArgMngr.add_workload('unravel_index', np.array([],dtype=_np.int64), (10, 3, 5))
    OpArgMngr.add_workload('unravel_index', np.array([3], dtype=_np.int32), (2,2))


def _add_workload_diag_indices_from():
    a = np.random.uniform(-4, 4, size=(4,4))
    OpArgMngr.add_workload('diag_indices_from', a)


def _add_workload_bincount():
    y = np.arange(4).astype(int)
    y1 = np.array([1, 5, 2, 4, 1], dtype=_np.int64)
    y2 = np.array((), dtype=_np.int8)
    w = np.array([0.2, 0.3, 0.5, 0.1])
    w1 = np.array([0.2, 0.3, 0.5, 0.1, 0.2])

    OpArgMngr.add_workload('bincount', y)
    OpArgMngr.add_workload('bincount', y1)
    OpArgMngr.add_workload('bincount', y, w)
    OpArgMngr.add_workload('bincount', y1, w1)
    OpArgMngr.add_workload('bincount', y1, w1, 8)
    OpArgMngr.add_workload('bincount', y, minlength=3)
    OpArgMngr.add_workload('bincount', y, minlength=8)
    OpArgMngr.add_workload('bincount', y2, minlength=0)
    OpArgMngr.add_workload('bincount', y2, minlength=5)


def _add_workload_cross():
    shapes = [
        # (a_shape, b_shape, (a_axis, b_axis, c_axis))
        ((2,), (2,), (-1, -1, -1)),
        ((1, 2), (1, 2), (-1, -1, -1)),
        ((2, 5, 4, 3), (5, 2, 4, 3), (0, 1, 2)),
        ((2, 5, 1, 3), (1, 2, 4, 3), (0, 1, 2)),

        ((2,), (3,), (-1, -1, -1)),
        ((1, 2,), (1, 3,), (-1, -1, -1)),
        ((6, 2, 5, 4), (6, 5, 3, 4), (1, 2, 0)),
        ((6, 2, 1, 4), (1, 5, 3, 4), (1, 2, 0)),

        ((3,), (2,), (-1, -1, -1)),
        ((1, 3,), (1, 2,), (-1, -1, -1)),
        ((6, 3, 5, 4), (6, 5, 2, 4), (1, 2, 0)),
        ((6, 3, 1, 4), (1, 5, 2, 4), (1, 2, 0)),

        ((3,), (3,), (-1, -1, -1)),
        ((1, 3,), (1, 3,), (-1, -1, -1)),
        ((6, 3, 5, 4), (6, 5, 3, 4), (1, 2, 0)),
        ((6, 3, 1, 4), (1, 5, 3, 4), (1, 2, 0)),
    ]
    dtypes = [np.float32, np.float64]
    for shape, dtype in itertools.product(shapes, dtypes):
        a_shape, b_shape, (a_axis, b_axis, c_axis) = shape
        a_np = _np.random.uniform(-10., 10., size=a_shape)
        b_np = _np.random.uniform(-10., 10., size=b_shape)
        a = np.array(a_np, dtype=dtype)
        b = np.array(b_np, dtype=dtype)
        OpArgMngr.add_workload('cross', a, b, axisa=a_axis, axisb=b_axis, axisc=c_axis)


def _add_workload_diag():
    def get_mat(n):
        data = _np.arange(n)
        data = _np.add.outer(data, data)
        return data

    A = np.array([[1, 2], [3, 4], [5, 6]])
    vals = (100 * np.arange(5)).astype('l')
    vals_c = (100 * np.array(get_mat(5)) + 1).astype('l')
    vals_f = _np.array((100 * get_mat(5) + 1), order='F', dtype='l')
    vals_f = np.array(vals_f)

    OpArgMngr.add_workload('diag', A, k=2)
    OpArgMngr.add_workload('diag', A, k=1)
    OpArgMngr.add_workload('diag', A, k=0)
    OpArgMngr.add_workload('diag', A, k=-1)
    OpArgMngr.add_workload('diag', A, k=-2)
    OpArgMngr.add_workload('diag', A, k=-3)
    OpArgMngr.add_workload('diag', vals, k=0)
    OpArgMngr.add_workload('diag', vals, k=2)
    OpArgMngr.add_workload('diag', vals, k=-2)
    OpArgMngr.add_workload('diag', vals_c, k=0)
    OpArgMngr.add_workload('diag', vals_c, k=2)
    OpArgMngr.add_workload('diag', vals_c, k=-2)
    OpArgMngr.add_workload('diag', vals_f, k=0)
    OpArgMngr.add_workload('diag', vals_f, k=2)
    OpArgMngr.add_workload('diag', vals_f, k=-2)


def _add_workload_diagonal():
    A = np.arange(12).reshape((3, 4))
    B = np.arange(8).reshape((2,2,2))

    OpArgMngr.add_workload('diagonal', A)
    OpArgMngr.add_workload('diagonal', A, offset=0)
    OpArgMngr.add_workload('diagonal', A, offset=-1)
    OpArgMngr.add_workload('diagonal', A, offset=1)
    OpArgMngr.add_workload('diagonal', B, offset=0)
    OpArgMngr.add_workload('diagonal', B, offset=1)
    OpArgMngr.add_workload('diagonal', B, offset=-1)
    OpArgMngr.add_workload('diagonal', B, 0, 1, 2)
    OpArgMngr.add_workload('diagonal', B, 0, 0, 1)
    OpArgMngr.add_workload('diagonal', B, offset=1, axis1=0, axis2=2)
    OpArgMngr.add_workload('diagonal', B, 0, 2, 1)


def _add_workload_median(array_pool):
    OpArgMngr.add_workload('median', array_pool['4x1'])
    OpArgMngr.add_workload('median', array_pool['4x1'], axis=0, keepdims=True)
    OpArgMngr.add_workload('median', np.array([[1, 2, 3], [4, 5, 6]]))
    OpArgMngr.add_workload('median', np.array([[1, 2, 3], [4, 5, 6]]), axis=0)
    OpArgMngr.add_workload('median', np.array([[1, 2, 3], [4, 5, 6]]), axis=1)


def _add_workload_quantile():
    x1 = np.arange(8) * 0.5
    x2 = np.arange(100.)
    q1 = np.array(0)
    q2 = np.array(1)
    q3 = np.array(0.5)
    q4 = np.array([0, 0.75, 0.25, 0.5, 1.0])
    q5 = 0.4

    OpArgMngr.add_workload('quantile', x1, q1)
    OpArgMngr.add_workload('quantile', x1, q2)
    OpArgMngr.add_workload('quantile', x1, q3)
    OpArgMngr.add_workload('quantile', x2, q4, interpolation="midpoint")
    OpArgMngr.add_workload('quantile', x2, q4, interpolation="nearest")
    OpArgMngr.add_workload('quantile', x2, q4, interpolation="lower")
    OpArgMngr.add_workload('quantile', x2, q5, interpolation="midpoint")
    OpArgMngr.add_workload('quantile', x2, q5, interpolation="nearest")
    OpArgMngr.add_workload('quantile', x2, q5, interpolation="lower")


def _add_workload_percentile():
    x1 = np.ones(5)
    q1 = np.array(5)
    x2 = np.array([[1, 1, 1],
                   [1, 1, 1],
                   [4, 4, 3],
                   [1, 1, 1],
                   [1, 1, 1]])
    q2 = np.array(60)
    x3 = np.arange(10)
    q3 = np.array([25, 50, 100])
    q4 = 65
    x4 = np.arange(11 * 2).reshape(11, 1, 2, 1)
    x5 = np.array([0, _np.nan])

    OpArgMngr.add_workload('percentile', x1, q1, None, None, None)
    OpArgMngr.add_workload('percentile', x1, q1, None, None, None, 'linear')
    OpArgMngr.add_workload('percentile', x2, q2, axis=0)
    OpArgMngr.add_workload('percentile', x3, q2, interpolation='linear')
    OpArgMngr.add_workload('percentile', x3, q2, interpolation='lower')
    OpArgMngr.add_workload('percentile', x3, q2, interpolation='higher')
    OpArgMngr.add_workload('percentile', x3, q2, interpolation='midpoint')
    OpArgMngr.add_workload('percentile', x3, q2, interpolation='nearest')
    OpArgMngr.add_workload('percentile', x3, q3)
    OpArgMngr.add_workload('percentile', x4, q2, axis=0)
    OpArgMngr.add_workload('percentile', x4, q2, axis=1)
    OpArgMngr.add_workload('percentile', x4, q4, axis=2)
    OpArgMngr.add_workload('percentile', x4, q4, axis=3)
    OpArgMngr.add_workload('percentile', x4, q2, axis=-1)
    OpArgMngr.add_workload('percentile', x4, q2, axis=-2)
    OpArgMngr.add_workload('percentile', x4, q4, axis=-3)
    OpArgMngr.add_workload('percentile', x4, q4, axis=-4)
    OpArgMngr.add_workload('percentile', x4, q2, axis=(1,2))
    OpArgMngr.add_workload('percentile', x4, q3, axis=(-2,-1))
    OpArgMngr.add_workload('percentile', x4, q2, axis=(1,2), keepdims=True)
    OpArgMngr.add_workload('percentile', x5, q2)
    OpArgMngr.add_workload('percentile', x5, q3)


def _add_workload_concatenate(array_pool):
    OpArgMngr.add_workload('concatenate', [array_pool['4x1'], array_pool['4x1']])
    OpArgMngr.add_workload('concatenate', [array_pool['4x1'], array_pool['4x1']], axis=1)
    OpArgMngr.add_workload('concatenate', [np.random.uniform(size=(3, 3))])
    OpArgMngr.add_workload('concatenate', (np.arange(4).reshape((2, 2)), np.arange(4).reshape((2, 2))))
    OpArgMngr.add_workload('concatenate', (np.arange(4),))
    OpArgMngr.add_workload('concatenate', (np.array(np.arange(4)),))
    OpArgMngr.add_workload('concatenate', (np.arange(4), np.arange(3)))
    OpArgMngr.add_workload('concatenate', (np.array(np.arange(4)), np.arange(3)))
    OpArgMngr.add_workload('concatenate', (np.arange(4), np.arange(3)), axis=0)
    OpArgMngr.add_workload('concatenate', (np.arange(4), np.arange(3)), axis=-1)
    a23 = np.random.uniform(size=(2, 3))
    a13 = np.random.uniform(size=(1, 3))
    OpArgMngr.add_workload('concatenate', (a23, a13))
    OpArgMngr.add_workload('concatenate', (a23, a13), axis=0)
    OpArgMngr.add_workload('concatenate', (a23.T, a13.T), axis=1)
    OpArgMngr.add_workload('concatenate', (a23.T, a13.T), axis=-1)
    res = np.arange(2*3*7).reshape((2, 3, 7))
    a0 = res[..., :4]
    a1 = res[..., 4:6]
    a2 = res[..., 6:]
    OpArgMngr.add_workload('concatenate', (a0, a1, a2), axis=2)
    OpArgMngr.add_workload('concatenate', (a0, a1, a2), axis=-1)
    OpArgMngr.add_workload('concatenate', (a0.T, a1.T, a2.T), axis=0)
    out = np.empty(4, dtype=np.float32)
    OpArgMngr.add_workload('concatenate', (np.array([1, 2]), np.array([3, 4])), out=out)
    OpArgMngr.add_workload('concatenate', [array_pool['4x1'], array_pool['4x1']], axis=None)
    OpArgMngr.add_workload('concatenate', (np.arange(4).reshape((2, 2)), np.arange(4).reshape((2, 2))), axis=None)
    OpArgMngr.add_workload('concatenate', (a23, a13), axis=None)


def _add_workload_append():

    def get_new_shape(shape, axis):
        shape_lst = list(shape)
        if axis is not None:
            shape_lst[axis] = _np.random.randint(0, 3)
        return tuple(shape_lst)

    for shape in [(0, 0), (2, 3), (2, 1, 3)]:
        for axis in [0, 1, None]:
            a = np.random.uniform(-1.0, 1.0, size=get_new_shape(shape, axis))
            b = np.random.uniform(-1.0, 1.0, size=get_new_shape(shape, axis))
            OpArgMngr.add_workload('append', a, b, axis=axis)

    OpArgMngr.add_workload('append', np.array([]), np.array([]))


def _add_workload_copy():
    OpArgMngr.add_workload('copy', np.random.uniform(size=(4, 1)))
    OpArgMngr.add_workload('copy', np.random.uniform(size=(2, 2)))
    OpArgMngr.add_workload('copy', np.random.uniform(size=(2,2)))


def _add_workload_expand_dims():
    OpArgMngr.add_workload('expand_dims', np.random.uniform(size=(4, 1)), -1)
    OpArgMngr.add_workload('expand_dims', np.random.uniform(size=(4, 1)) > 0.5, -1)
    for axis in range(-5, 4):
        OpArgMngr.add_workload('expand_dims', np.empty((2, 3, 4, 5)), axis)


def _add_workload_split():
    OpArgMngr.add_workload('split', np.random.uniform(size=(4, 1)), 2)
    OpArgMngr.add_workload('split', np.arange(10), 2)
    OpArgMngr.add_workload('split', np.random.uniform(size=(10, 10, 3)), 3, -1)
    assertRaises(ValueError, np.split, np.arange(10), 3)


def _add_workload_array_split():
    a = np.arange(10)
    b = np.array([np.arange(10), np.arange(10)])

    for i in range(1, 12):
        OpArgMngr.add_workload('array_split', a, i)
    OpArgMngr.add_workload('array_split', b, 3, axis=0)
    OpArgMngr.add_workload('array_split', b, [0, 1, 2], axis=0)
    OpArgMngr.add_workload('array_split', b, 3, axis=-1)
    OpArgMngr.add_workload('array_split', b, 3)


def _add_workload_hsplit():
    a = np.array([1, 2, 3, 4])
    OpArgMngr.add_workload('hsplit', a, 2)
    b = np.array([[1, 2, 3, 4], [1, 2, 3, 4]])
    OpArgMngr.add_workload('hsplit', b, 2)


def _add_workload_vsplit():
    assertRaises(ValueError, np.vsplit, np.array([1, 2, 3, 4]), 2)
    a = np.array([[1, 2, 3, 4], [1, 2, 3, 4]])
    OpArgMngr.add_workload('vsplit', a, 2)
    assertRaises(ValueError, np.vsplit, np.array(1), 2)


def _add_workload_dsplit():
    a = np.array([[[1, 2, 3, 4], [1, 2, 3, 4]],
                  [[1, 2, 3, 4], [1, 2, 3, 4]]])
    OpArgMngr.add_workload('dsplit', a, 2)
    assertRaises(ValueError, np.dsplit, np.array(1), 2)
    assertRaises(ValueError, np.dsplit, np.array([1, 2, 3, 4]), 2)
    assertRaises(ValueError, np.dsplit, np.array([[1, 2, 3, 4], [1, 2, 3, 4]]), 2)


def _add_workload_squeeze():
    OpArgMngr.add_workload('squeeze', np.random.uniform(size=(4, 1)))
    OpArgMngr.add_workload('squeeze', np.random.uniform(size=(20, 10, 10, 1, 1)))
    OpArgMngr.add_workload('squeeze', np.random.uniform(size=(20, 1, 10, 1, 20)))
    OpArgMngr.add_workload('squeeze', np.random.uniform(size=(1, 1, 20, 10)))
    OpArgMngr.add_workload('squeeze', np.array([[[1.5]]]))


def _add_workload_std():
    OpArgMngr.add_workload('std', np.random.uniform(size=(4, 1)))
    A = np.array([[1, 2, 3], [4, 5, 6]])
    OpArgMngr.add_workload('std', A)
    OpArgMngr.add_workload('std', A, 0)
    OpArgMngr.add_workload('std', A, 1)
    OpArgMngr.add_workload('std', np.array([1, -1, 1, -1]))
    OpArgMngr.add_workload('std', np.array([1, -1, 1, -1]), ddof=1)
    OpArgMngr.add_workload('std', np.array([1, -1, 1, -1]), ddof=2)
    OpArgMngr.add_workload('std', np.arange(10), out=np.array(0.))


def _add_workload_swapaxes():
    OpArgMngr.add_workload('swapaxes', np.random.uniform(size=(4, 1)), 0, 1)
    OpArgMngr.add_workload('swapaxes', np.array([[[0, 1], [2, 3]], [[4, 5], [6, 7]]]), 0, 2)
    a = np.arange(1*2*3*4).reshape(1, 2, 3, 4).copy()
    b = a.copy()
    # no AxisError defined in mxnet numpy
    # assertRaises(np.AxisError, np.swapaxes, -5, 0)
    for i in range(-4, 4):
        for j in range(-4, 4):
            for src in (a, b):
                OpArgMngr.add_workload('swapaxes', src, i, j)


def _add_workload_tensordot():
    OpArgMngr.add_workload('tensordot', np.random.uniform(size=(4, 1)), np.random.uniform(size=(4, 1)))
    OpArgMngr.add_workload('tensordot', np.random.uniform(size=(3, 0)), np.random.uniform(size=(0, 4)), (1, 0))
    OpArgMngr.add_workload('tensordot', np.array(1), np.array(1), ([], []))


def _add_workload_tile():
    OpArgMngr.add_workload('tile', np.random.uniform(size=(4, 1)), 2)
    a = np.array([0, 1, 2])
    b = np.array([[1, 2], [3, 4]])
    OpArgMngr.add_workload('tile', a, 2)
    OpArgMngr.add_workload('tile', a, (2, 2))
    OpArgMngr.add_workload('tile', a, (1, 2))
    OpArgMngr.add_workload('tile', b, 2)
    OpArgMngr.add_workload('tile', b, (2, 1))
    OpArgMngr.add_workload('tile', b, (2, 2))
    OpArgMngr.add_workload('tile', np.arange(5), 1)
    OpArgMngr.add_workload('tile', np.array([[], []]), 2)
    OpArgMngr.add_workload('tile', np.array([[[]]]), (3, 2, 5))
    reps = [(2,), (1, 2), (2, 1), (2, 2), (2, 3, 2), (3, 2)]
    shape = [(3,), (2, 3), (3, 4, 3), (3, 2, 3), (4, 3, 2, 4), (2, 2)]
    for s in shape:
        b = np.random.randint(0, 10, size=s)
        for _ in reps:
            # RuntimeError to be tracked
            # where s = (3, 4, 3), r = (2, 3, 2)
            # OpArgMngr.add_workload('tile', b, r)
            pass


def _add_workload_transpose():
    OpArgMngr.add_workload('transpose', np.random.uniform(size=(4, 1)))
    OpArgMngr.add_workload('transpose', np.array([[]]))
    OpArgMngr.add_workload('transpose', np.array([[1, 2]]))
    OpArgMngr.add_workload('transpose', np.array([[1, 2, 3], [4, 5, 6]]))
    OpArgMngr.add_workload('transpose', np.array([[1, 2], [3, 4], [5, 6]]), (1, 0))
    OpArgMngr.add_workload('transpose', np.array([[1, 2], [3, 4]]))


def _add_workload_linalg_norm():
    OpArgMngr.add_workload('linalg.norm', np.random.uniform(size=(4, 1)))
    for dt in ["float64", "float32"]:
        OpArgMngr.add_workload('linalg.norm', np.array([], dtype=dt))
        OpArgMngr.add_workload('linalg.norm', np.array([np.array([]), np.array([])], dtype=dt))
        for v in ([1, 2, 3, 4], [-1, -2, -3, -4], [-1, 2, -3, 4]):
            OpArgMngr.add_workload('linalg.norm', np.array(v, dtype=dt))
        A = np.array([[1, 2, 3], [4, 5, 6]], dtype=dt)
        [OpArgMngr.add_workload('linalg.norm', A[:, k]) for k in range(A.shape[1])]
        OpArgMngr.add_workload('linalg.norm', A, axis=0)
        [OpArgMngr.add_workload('linalg.norm', A[k, :]) for k in range(A.shape[0])]
        OpArgMngr.add_workload('linalg.norm', A, axis=1)
        B = np.arange(1, 25).reshape(2, 3, 4).astype(dt)
        for axis in itertools.combinations(range(-B.ndim, B.ndim), 2):
            row_axis, col_axis = axis
            if row_axis < 0:
                row_axis += B.ndim
            if col_axis < 0:
                col_axis += B.ndim
            if row_axis == col_axis:
                # improper assertion behavior
                # assertRaises(ValueError, np.linalg.norm, B, axis=axis)
                pass
            else:
                OpArgMngr.add_workload('linalg.norm', B, axis=axis)
                k_index = B.ndim - row_axis - col_axis
                for k in range(B.shape[k_index]):
                    if row_axis < col_axis:
                        OpArgMngr.add_workload('linalg.norm', np.take(B[:], np.array(k), axis=k_index))
                    else:
                        OpArgMngr.add_workload('linalg.norm', np.take(B[:], np.array(k), axis=k_index).T)
        A = np.arange(1, 25, dtype=dt).reshape(2, 3, 4)
        OpArgMngr.add_workload('linalg.norm', A, ord=None, axis=None)
        OpArgMngr.add_workload('linalg.norm', A, ord=None, axis=None, keepdims=True)
        for k in range(A.ndim):
            OpArgMngr.add_workload('linalg.norm', A, axis=k)
            OpArgMngr.add_workload('linalg.norm', A, axis=k, keepdims=True)
        for k in itertools.permutations(range(A.ndim), 2):
            OpArgMngr.add_workload('linalg.norm', A, axis=k)
            OpArgMngr.add_workload('linalg.norm', A, axis=k, keepdims=True)
        OpArgMngr.add_workload('linalg.norm', np.array([[]], dtype=dt))
        A = np.array([[1, 3], [5, 7]], dtype=dt)
        OpArgMngr.add_workload('linalg.norm', A, 2)
        OpArgMngr.add_workload('linalg.norm', A, -2)
        OpArgMngr.add_workload('linalg.norm', A, 'nuc')
        A = (1 / 10) * np.array([[1, 2, 3], [6, 0, 5], [3, 2, 1]], dtype=dt)
        OpArgMngr.add_workload('linalg.norm', A)
        OpArgMngr.add_workload('linalg.norm', A, 'fro')
        OpArgMngr.add_workload('linalg.norm', A, 1)
        OpArgMngr.add_workload('linalg.norm', A, -1)
    for dt in [np.float32, np.float64]:
        OpArgMngr.add_workload('linalg.norm', np.array([[1, 0, 1], [0, 1, 1]], dtype=dt))
        OpArgMngr.add_workload('linalg.norm', np.array([[1, 0, 1], [0, 1, 1]], dtype=dt), 'fro')


def _add_workload_linalg_cholesky():
    shapes = [(1, 1), (2, 2), (3, 3), (50, 50), (3, 10, 10)]
    dtypes = (np.float32, np.float64)

    with random_seed(1):
        for shape, dtype in itertools.product(shapes, dtypes):
            a = _np.random.randn(*shape)

        t = list(range(len(shape)))
        t[-2:] = -1, -2

        a = _np.matmul(a.transpose(t).conj(), a)

        OpArgMngr.add_workload('linalg.cholesky', np.array(a, dtype=dtype))

    # test_0_size
    for dtype in dtypes:
        a = np.zeros((0, 1, 1))
        OpArgMngr.add_workload('linalg.cholesky', np.array(a, dtype=dtype))
        a = np.zeros((1, 0, 0))
        OpArgMngr.add_workload('linalg.cholesky', np.array(a, dtype=dtype))


def _add_workload_linalg_qr():
    A = np.array([[0, 1], [1, 1], [1, 1], [2, 1]])
    OpArgMngr.add_workload('linalg.qr', A)
    # default mode in numpy is 'reduced'
    OpArgMngr.add_workload('linalg.qr', A, mode='reduced')


def _add_workload_linalg_inv():
    OpArgMngr.add_workload('linalg.inv', np.array(_np.ones((0, 0)), dtype=np.float32))
    OpArgMngr.add_workload('linalg.inv', np.array(_np.ones((0, 1, 1)), dtype=np.float64))


def _add_workload_linalg_solve():
    shapes = [(0,0), (1,1), (5,5), (6,6), (3,5,5), (3,0,0), (2,5,5), (0,5,5), (2,3,4,4)]
    nrhs = (0, 1, 2, 3)
    dtypes = (np.float32, np.float64)
    for dtype, shape in itertools.product(dtypes, shapes):
        a = _np.random.rand(*shape)
        shape_b = list(shape)
        shape_b[-1] = 1
        x = _np.random.rand(*shape_b)
        b = _np.matmul(a, x)
        shape_b.pop()
        b = b.reshape(shape_b)
        OpArgMngr.add_workload('linalg.solve', np.array(a, dtype=dtype), np.array(b, dtype=dtype))
        for nrh in nrhs:
            shape_b = list(shape)
            shape_b[-1] = nrh
            x = _np.random.rand(*shape_b)
            b = _np.matmul(a, x)
            OpArgMngr.add_workload('linalg.solve', np.array(a, dtype=dtype), np.array(b, dtype=dtype))


def _add_workload_linalg_det():
    OpArgMngr.add_workload('linalg.det', np.array(_np.ones((2, 2)), dtype=np.float32))
    OpArgMngr.add_workload('linalg.det', np.array(_np.ones((0, 1, 1)), dtype=np.float64))


def _add_workload_linalg_tensorinv():
    shapes = [
        (1, 20, 4, 5),
        (2, 2, 10, 4, 5),
        (2, 12, 5, 3, 4, 5),
        (3, 2, 3, 4, 24)
    ]
    dtypes = (np.float32, np.float64)
    for dtype, shape in itertools.product(dtypes, shapes):
        ind = shape[0]
        prod_front = 1
        prod_back = 1
        for k in shape[1:ind + 1]:
            prod_front *= k
        for k in shape[1 + ind:]:
            prod_back *= k
        a_shape = (prod_back, prod_front)
        a = _np.random.randn(*a_shape)
        if prod_back == prod_front:
            if _np.allclose(_np.dot(a, _np.linalg.inv(a)), _np.eye(prod_front)):
                a_shape = shape[1:]
                a = a.reshape(a_shape)
                OpArgMngr.add_workload('linalg.tensorinv', np.array(a, dtype=dtype), ind)


def _add_workload_linalg_tensorsolve():
    shapes = [
        # a_shape.ndim <= 6
        # (a_shape, b_shape, axes)
        ((1, 1), (1,), None),
        ((1, 1), (1, 1, 1, 1, 1), None),
        ((4, 4), (4,), None),
        ((2, 3, 3, 4, 2), (3, 4), (0, 2, 4)),
        ((1, 3, 3, 4, 4), (1, 3, 4), (1, 3)),
        ((1, 4, 1, 12, 3), (1, 2, 1, 2, 1, 3, 1), (1, 2, 4)),
    ]
    dtypes = (np.float32, np.float64)
    for dtype in dtypes:
        for a_shape, b_shape, axes in shapes:
            a_ndim = len(a_shape)
            b_ndim = len(b_shape)
            a_trans_shape = list(a_shape)
            a_axes = list(range(0, a_ndim))
            if axes is not None:
                for k in axes:
                    a_axes.remove(k)
                    a_axes.insert(a_ndim, k)
                for k in range(a_ndim):
                    a_trans_shape[k] = a_shape[a_axes[k]]
            x_shape = a_trans_shape[-(a_ndim - b_ndim):]
            prod = 1
            for k in x_shape:
                prod *= k
            if prod * prod != _np.prod(a_shape):
                raise ValueError("a is not square")
            if prod != _np.prod(b_shape):
                raise ValueError("a's shape and b's shape dismatch")
            mat_shape = (prod, prod)
            a_trans_shape = tuple(a_trans_shape)
            x_shape = tuple(x_shape)

            a_np = _np.eye(prod)
            shape = mat_shape
            while 1:
                # generate well-conditioned matrices with small eigenvalues
                D = _np.diag(_np.random.uniform(-1.0, 1.0, shape[-1]))
                I = _np.eye(shape[-1]).reshape(shape)
                v = _np.random.uniform(-1., 1., shape[-1]).reshape(shape[:-1] + (1,))
                v = v / _np.linalg.norm(v, axis=-2, keepdims=True)
                v_T = _np.swapaxes(v, -1, -2)
                U = I - 2 * _np.matmul(v, v_T)
                a = _np.matmul(U, D)
                if (_np.linalg.cond(a, 2) < 4):
                    a_np = a.reshape(a_trans_shape)
                    break
            x_np = _np.random.randn(*x_shape)
            b_np = _np.tensordot(a_np, x_np, axes=len(x_shape))
            a_origin_axes = list(range(a_np.ndim))
            if axes is not None:
                for k in range(a_np.ndim):
                    a_origin_axes[a_axes[k]] = k
            a_np = a_np.transpose(a_origin_axes)
            OpArgMngr.add_workload('linalg.tensorsolve', np.array(a_np, dtype=dtype), np.array(b_np, dtype=dtype), axes)


def _add_workload_linalg_pinv():
    shapes = [
        ((1, 1), ()),
        ((5, 5), ()),
        ((5, 6), ()),
        ((6, 5), ()),
        ((2, 3, 3), (1,)),
        ((4, 6, 5), (4,)),
        ((2, 2, 3, 4), (2, 2)),
    ]
    dtypes = (np.float32, np.float64)
    for dtype in dtypes:
        for a_shape, rcond_shape in shapes:
            hermitian = False
            a_np = _np.random.uniform(-10.0, 10.0, a_shape)
            a_np = _np.array(a_np, dtype=dtype)
            rcond_np = _np.random.uniform(0., 0.1, rcond_shape)
            rcond_np = _np.array(rcond_np, dtype=dtype)
            OpArgMngr.add_workload('linalg.pinv', np.array(a_np, dtype=dtype), np.array(rcond_np, dtype=dtype), hermitian)


def _add_workload_linalg_lstsq():
    shapes = [
        ((0, 0), (0,)),
        ((0, 0), (0, 0)),
        ((4, 0), (4,)),
        ((4, 0), (4, 2)),
        ((0, 2), (0, 4)),
        ((4, 2), (4, 0)),
        ((0, 0), (0, 4)),
        ((0, 2), (0, 0)),
        ((4, 0), (4, 0)),
        ((4, 2), (4,)),
        ((4, 2), (4, 3)),
        ((4, 6), (4, 3)),
    ]
    rconds = [None, "random", "warn"]
    dtypes = (np.float32, np.float64)
    for dtype, rcond in itertools.product(dtypes, rconds):
        for a_shape, b_shape in shapes:
            if rcond == "random":
                rcond = _np.random.uniform(100, 200)
            if rcond == "warn":
                rcond = -1
            a_np = _np.random.uniform(-10.0, 10.0, a_shape)
            b_np = _np.random.uniform(-10.0, 10.0, b_shape)
            a = np.array(a_np, dtype=dtype)
            b = np.array(b_np, dtype=dtype)
            OpArgMngr.add_workload('linalg.lstsq', a, b, rcond)


def _add_workload_linalg_eigvals():
    OpArgMngr.add_workload('linalg.eigvals', np.array(_np.diag((0, 0)), dtype=np.float64))
    OpArgMngr.add_workload('linalg.eigvals', np.array(_np.diag((1, 1)), dtype=np.float64))
    OpArgMngr.add_workload('linalg.eigvals', np.array(_np.diag((2, 2)), dtype=np.float64))


def _add_workload_linalg_eig():
    OpArgMngr.add_workload('linalg.eig', np.array(_np.diag((0, 0)), dtype=np.float64))
    OpArgMngr.add_workload('linalg.eig', np.array(_np.diag((1, 1)), dtype=np.float64))
    OpArgMngr.add_workload('linalg.eig', np.array(_np.diag((2, 2)), dtype=np.float64))


def _add_workload_linalg_eigvalsh():
    OpArgMngr.add_workload('linalg.eigvalsh', np.array(_np.diag((0, 0)), dtype=np.float64))
    OpArgMngr.add_workload('linalg.eigvalsh', np.array(_np.diag((1, 1)), dtype=np.float64))
    OpArgMngr.add_workload('linalg.eigvalsh', np.array(_np.diag((2, 2)), dtype=np.float64))


def _add_workload_linalg_eigh():
    OpArgMngr.add_workload('linalg.eigh', np.array(_np.diag((0, 0)), dtype=np.float64))
    OpArgMngr.add_workload('linalg.eigh', np.array(_np.diag((1, 1)), dtype=np.float64))
    OpArgMngr.add_workload('linalg.eigh', np.array(_np.diag((2, 2)), dtype=np.float64))


def _add_workload_linalg_slogdet():
    OpArgMngr.add_workload('linalg.slogdet', np.array(_np.ones((2, 2)), dtype=np.float32))
    OpArgMngr.add_workload('linalg.slogdet', np.array(_np.ones((0, 1, 1)), dtype=np.float64))


def _add_workload_trace():
    OpArgMngr.add_workload('trace', np.random.uniform(size=(4, 1)))
    OpArgMngr.add_workload('trace', np.random.uniform(size=(3, 2)))


def _add_workload_tril():
    OpArgMngr.add_workload('tril', np.random.uniform(size=(4, 1)))
    for dt in ['float16', 'float32', 'float64', 'int32', 'int64', 'int8', 'uint8']:
        OpArgMngr.add_workload('tril', np.ones((2, 2), dtype=dt))
        a = np.array([
            [[1, 1], [1, 1]],
            [[1, 1], [1, 0]],
            [[1, 1], [0, 0]],
        ], dtype=dt)
        OpArgMngr.add_workload('tril', a)
        arr = np.array([[1, 1, _np.inf],
                        [1, 1, 1],
                        [_np.inf, 1, 1]])
        OpArgMngr.add_workload('tril', arr)
        OpArgMngr.add_workload('tril', np.zeros((3, 3), dtype=dt))
    import mxnet as mx
    assertRaises(mx.MXNetError, np.tril, 10)
    assertRaises(mx.MXNetError, np.tril, 2, 10)


def _add_workload_triu():
    OpArgMngr.add_workload('triu', np.random.uniform(size=(4, 1)))
    for dt in ['float16', 'float32', 'float64', 'int32', 'int64', 'int8', 'uint8']:
        OpArgMngr.add_workload('triu', np.ones((2, 2), dtype=dt))
        a = np.array([
            [[1, 1], [1, 1]],
            [[1, 1], [1, 0]],
            [[1, 1], [0, 0]],
        ], dtype=dt)
        OpArgMngr.add_workload('triu', a)
        arr = np.array([[1, 1, _np.inf],
                        [1, 1, 1],
                        [_np.inf, 1, 1]])
        OpArgMngr.add_workload('triu', arr)
        OpArgMngr.add_workload('triu', np.zeros((3, 3), dtype=dt))


def _add_workload_einsum():
    chars = 'abcdefghij'
    sizes = [2, 3, 4, 5, 4, 3, 2, 6, 5, 4]
    size_dict = dict(zip(chars, sizes))

    configs = [
        # test_einsum_broadcast
        ('ij...,j...->ij...', [(2, 3, 4), (3,)]),
        ('ij...,...j->ij...', [(2, 3, 4), (3,)]),
        ('ij...,j->ij...', [(2, 3, 4), (3,)]),
        ('cl, cpx->lpx', [(2, 3), (2, 3, 2731)]),
        ('aabb->ab', [(5, 5, 5, 5)]),
        ('mi,mi,mi->m', [(5, 5), (5, 5), (5, 5)]),
        ('a,ab,abc->abc', None),
        ('a,b,ab->ab', None),
        ('ea,fb,gc,hd,abcd->efgh', None),
        ('ea,fb,abcd,gc,hd->efgh', None),
        ('abcd,ea,fb,gc,hd->efgh', None),
        # test_complex
        ('acdf,jbje,gihb,hfac,gfac,gifabc,hfac', None),
        ('acdf,jbje,gihb,hfac,gfac,gifabc,hfac', None),
        ('cd,bdhe,aidb,hgca,gc,hgibcd,hgac', None),
        ('abhe,hidj,jgba,hiab,gab', None),
        ('bde,cdh,agdb,hica,ibd,hgicd,hiac', None),
        ('chd,bde,agbc,hiad,hgc,hgi,hiad', None),
        ('chd,bde,agbc,hiad,bdi,cgh,agdb', None),
        ('bdhe,acad,hiab,agac,hibd', None),
        # test_collapse
        ('ab,ab,c->', None),
        ('ab,ab,c->c', None),
        ('ab,ab,cd,cd->', None),
        ('ab,ab,cd,cd->ac', None),
        ('ab,ab,cd,cd->cd', None),
        ('ab,ab,cd,cd,ef,ef->', None),
        # test_inner_product
        ('ab,ab', None),
        ('ab,ba', None),
        ('abc,abc', None),
        ('abc,bac', None),
        ('abc,cba', None),
        # test_random_cases
        ('aab,fa,df,ecc->bde', None),
        ('ecb,fef,bad,ed->ac', None),
        ('bcf,bbb,fbf,fc->', None),
        ('bb,ff,be->e', None),
        ('bcb,bb,fc,fff->', None),
        ('fbb,dfd,fc,fc->', None),
        ('afd,ba,cc,dc->bf', None),
        ('adb,bc,fa,cfc->d', None),
        ('bbd,bda,fc,db->acf', None),
        ('dba,ead,cad->bce', None),
        ('aef,fbc,dca->bde', None),
        # test_broadcasting_dot_cases
        ('ijk,kl,jl', [(1, 5, 4), (4, 6), (5, 6)]),
        ('ijk,kl,jl,i->i', [(1, 5, 4), (4, 6), (5, 6), (10)]),
        ('abjk,kl,jl', [(1, 1, 5, 4), (4, 6), (5, 6)]),
        ('abjk,kl,jl,ab->ab', [(1, 1, 5, 4), (4, 6), (5, 6), (7, 7)]),
        ('obk,ijk->ioj', [(2, 4, 8), (2, 4, 8)]),
    ]
    # check_einsum_sums
    configs.extend([('i->', [(i,)]) for i in range(1, 17)])
    configs.extend([('...i->...', [(2, 3, i,)]) for i in range(1, 17)])
    configs.extend([('i...->...', [(2, i,)]) for i in range(1, 17)])
    configs.extend([('i...->...', [(2, 3, i,)]) for i in range(1, 17)])
    configs.extend([('ii', [(i, i,)]) for i in range(1, 17)])
    configs.extend([('..., ...', [(3, i,), (2, 3, i,)]) for i in range(1, 17)])
    configs.extend([('...i, ...i', [(2, 3, i,), (i,)]) for i in range(1, 17)])
    configs.extend([('i..., i...', [(i, 3, 2,), (i,)]) for i in range(1, 11)])
    configs.extend([('i, j', [(3,), (i,)]) for i in range(1, 17)])
    configs.extend([('ij, j', [(4, i), (i,)]) for i in range(1, 17)])
    configs.extend([('ji, j', [(i, 4), (i,)]) for i in range(1, 17)])
    configs.extend([('ij, jk', [(4, i), (i, 6)]) for i in range(1, 8)])
    configs.extend([
        ('ij,jk,kl', [(3, 4), (4, 5), (5, 6)]),
        ('ijk, jil -> kl', [(3, 4, 5), (4, 3, 2)]),
        ('i, i, i -> i', [(8,), (8,), (8,)]),
        (',i->', [(), (9,)]),
        ('i,->', [(9,), ()]),
    ])
    configs.extend([('...,...', [(n,), (n,)]) for n in range(1, 25)])
    configs.extend([('i,i', [(n,), (n,)]) for n in range(1, 25)])
    configs.extend([('i,->i', [(n,), ()]) for n in range(1, 25)])
    configs.extend([(',i->i', [(), (n,)]) for n in range(1, 25)])
    configs.extend([('i,->', [(n,), ()]) for n in range(1, 25)])
    configs.extend([(',i->', [(), (n,)]) for n in range(1, 25)])
    configs.extend([('...,...', [(n - 1,), (n - 1,)]) for n in range(1, 25)])
    configs.extend([('i,i', [(n - 1,), (n - 1,)]) for n in range(1, 25)])
    configs.extend([('i,->i', [(n - 1,), ()]) for n in range(1, 25)])
    configs.extend([(',i->i', [(), (n - 1,)]) for n in range(1, 25)])
    configs.extend([('i,->', [(n - 1,), ()]) for n in range(1, 25)])
    configs.extend([(',i->', [(), (n - 1,)]) for n in range(1, 25)])

    for optimize in [False, True]:
        for config in configs:
            subscripts, args = config
            if args is None:
                args = []
                terms = subscripts.split('->')[0].split(',')
                for term in terms:
                    dims = [size_dict[x] for x in term]
                    args.append(np.random.uniform(size=dims))
            else:
                args = [np.random.uniform(size=arg) for arg in args]
            OpArgMngr.add_workload('einsum', subscripts, *args, optimize=optimize)


def _add_workload_expm1():
    OpArgMngr.add_workload('expm1', np.random.uniform(size=(4, 1)))
    OpArgMngr.add_workload('expm1', np.random.uniform(size=(1, 1)))
    OpArgMngr.add_workload('expm1', np.array([_np.inf]))
    OpArgMngr.add_workload('expm1', np.array([-_np.inf]))
    OpArgMngr.add_workload('expm1', np.array([0.]))
    OpArgMngr.add_workload('expm1', np.array([-0.]))


def _add_workload_argmax():
    OpArgMngr.add_workload('argmax', np.random.uniform(size=(4, 5, 6, 7, 8)), 0)
    OpArgMngr.add_workload('argmax', np.random.uniform(size=(4, 5, 6, 7, 8)), 1)
    OpArgMngr.add_workload('argmax', np.random.uniform(size=(4, 5, 6, 7, 8)), 2)
    OpArgMngr.add_workload('argmax', np.random.uniform(size=(4, 5, 6, 7, 8)), 3)
    OpArgMngr.add_workload('argmax', np.random.uniform(size=(4, 5, 6, 7, 8)), 4)
    # OpArgMngr.add_workload('argmax', np.array([0, 1, 2, 3, _np.nan]))
    # OpArgMngr.add_workload('argmax', np.array([0, 1, 2, _np.nan, 3]))
    # OpArgMngr.add_workload('argmax', np.array([_np.nan, 0, 1, 2, 3]))
    # OpArgMngr.add_workload('argmax', np.array([_np.nan, 0, _np.nan, 2, 3]))
    OpArgMngr.add_workload('argmax', np.array([False, False, False, False, True]))
    OpArgMngr.add_workload('argmax', np.array([False, False, False, True, False]))
    OpArgMngr.add_workload('argmax', np.array([True, False, False, False, False]))
    OpArgMngr.add_workload('argmax', np.array([True, False, True, False, False]))


def _add_workload_argmin():
    OpArgMngr.add_workload('argmin', np.random.uniform(size=(4, 5, 6, 7, 8)), 0)
    OpArgMngr.add_workload('argmin', np.random.uniform(size=(4, 5, 6, 7, 8)), 1)
    OpArgMngr.add_workload('argmin', np.random.uniform(size=(4, 5, 6, 7, 8)), 2)
    OpArgMngr.add_workload('argmin', np.random.uniform(size=(4, 5, 6, 7, 8)), 3)
    OpArgMngr.add_workload('argmin', np.random.uniform(size=(4, 5, 6, 7, 8)), 4)
    # OpArgMngr.add_workload('argmin', np.array([0, 1, 2, 3, _np.nan]))
    # OpArgMngr.add_workload('argmin', np.array([0, 1, 2, _np.nan, 3]))
    # OpArgMngr.add_workload('argmin', np.array([_np.nan, 0, 1, 2, 3]))
    # OpArgMngr.add_workload('argmin', np.array([_np.nan, 0, _np.nan, 2, 3]))
    OpArgMngr.add_workload('argmin', np.array([False, False, False, False, True]))
    OpArgMngr.add_workload('argmin', np.array([False, False, False, True, False]))
    OpArgMngr.add_workload('argmin', np.array([True, False, False, False, False]))
    OpArgMngr.add_workload('argmin', np.array([True, False, True, False, False]))


def _add_workload_around():
    OpArgMngr.add_workload('around', np.array([1.56, 72.54, 6.35, 3.25]), decimals=1)


def _add_workload_round():
    OpArgMngr.add_workload('round', np.array([1.56, 72.54, 6.35, 3.25]), decimals=1)


def _add_workload_round_():
    OpArgMngr.add_workload('round_', np.array([1.56, 72.54, 6.35, 3.25]), decimals=1)


def _add_workload_argsort():
    for dtype in [np.int32, np.float32]:
        a = np.arange(101, dtype=dtype)
        OpArgMngr.add_workload('argsort', a)
    OpArgMngr.add_workload('argsort', np.array([[3, 2], [1, 0]]), 1)
    OpArgMngr.add_workload('argsort', np.array([[3, 2], [1, 0]]), 0)
    a = np.ones((3, 2, 1, 0))
    for axis in range(-a.ndim, a.ndim):
        OpArgMngr.add_workload('argsort', a, axis)


def _add_workload_sort():
    OpArgMngr.add_workload('sort', np.random.uniform(0, 100), axis=None)
    OpArgMngr.add_workload('sort', np.random.uniform(0, 100, size=()), axis=None)
    OpArgMngr.add_workload('sort', np.random.uniform(0, 100, size=(2, 3, 4)), axis=None)
    OpArgMngr.add_workload('sort', np.random.uniform(0, 100, size=(4, 3, 0)), axis=None)
    OpArgMngr.add_workload('sort', np.random.randint(0, 100, size=(2, 3, 4)), axis=-1)
    OpArgMngr.add_workload('sort', np.random.randint(0, 100, size=(4, 3, 5)), axis=-1, kind='mergesort')
    OpArgMngr.add_workload('sort', np.random.randint(0, 100, size=(2, 3, 4)), axis=None, kind='quicksort')
    OpArgMngr.add_workload('sort', np.random.uniform(0, 100, size=(4, 3, 0)))


def _add_workload_broadcast_arrays(array_pool):
    OpArgMngr.add_workload('broadcast_arrays', array_pool['4x1'], array_pool['1x2'])


def _add_workload_broadcast_to():
    OpArgMngr.add_workload('broadcast_to', np.array(0), (0,))
    OpArgMngr.add_workload('broadcast_to', np.array(0), (1,))
    OpArgMngr.add_workload('broadcast_to', np.array(0), (3,))
    OpArgMngr.add_workload('broadcast_to', np.ones(1), (1,))
    OpArgMngr.add_workload('broadcast_to', np.ones(1), (2,))
    OpArgMngr.add_workload('broadcast_to', np.ones(1), (1, 2, 3))
    OpArgMngr.add_workload('broadcast_to', np.arange(3), (3,))
    OpArgMngr.add_workload('broadcast_to', np.arange(3), (1, 3))
    OpArgMngr.add_workload('broadcast_to', np.arange(3), (2, 3))
    OpArgMngr.add_workload('broadcast_to', np.ones(0), 0)
    OpArgMngr.add_workload('broadcast_to', np.ones(1), 1)
    OpArgMngr.add_workload('broadcast_to', np.ones(1), 2)
    OpArgMngr.add_workload('broadcast_to', np.ones(1), (0,))
    OpArgMngr.add_workload('broadcast_to', np.ones((1, 2)), (0, 2))
    OpArgMngr.add_workload('broadcast_to', np.ones((2, 1)), (2, 0))


def _add_workload_clip():
    OpArgMngr.add_workload('clip', (np.random.normal(size=(1000,)) * 1024).astype("float"), -12.8, 100.2)
    OpArgMngr.add_workload('clip', (np.random.normal(size=(1000,)) * 1024).astype("float"), 0, 0)
    OpArgMngr.add_workload('clip', (np.random.normal(size=(1000,)) * 1024).astype("int"), -120, 100)
    OpArgMngr.add_workload('clip', (np.random.normal(size=(1000,)) * 1024).astype("int"), 0.0, 2.0)
    OpArgMngr.add_workload('clip', (np.random.normal(size=(1000,)) * 1024).astype("int"), 0, 0)
    OpArgMngr.add_workload('clip', (np.random.normal(size=(1000,)) * 1024).astype("uint8"), 0, 0)
    OpArgMngr.add_workload('clip', (np.random.normal(size=(1000,)) * 1024).astype("uint8"), 0.0, 2.0)
    OpArgMngr.add_workload('clip', (np.random.normal(size=(1000,)) * 1024).astype("uint8"), -120, 100)
    # OpArgMngr.add_workload('clip', np.random.normal(size=(1000,)), np.zeros((1000,))+0.5, 1)
    # OpArgMngr.add_workload('clip', np.random.normal(size=(1000,)), 0, np.zeros((1000,))+0.5)
    # OpArgMngr.add_workload('clip', np.array([0, 1, 2, 3, 4, 5, 6, 7]), 3)
    # OpArgMngr.add_workload('clip', np.array([0, 1, 2, 3, 4, 5, 6, 7]), a_min=3)
    # OpArgMngr.add_workload('clip', np.array([0, 1, 2, 3, 4, 5, 6, 7]), a_max=4)
    OpArgMngr.add_workload('clip', np.array([-2., _np.nan, 0.5, 3., 0.25, _np.nan]), -1, 1)


def _add_workload_cumsum():
    for ctype in _DTYPES:
        OpArgMngr.add_workload('cumsum', np.array([1, 2, 10, 11, 6, 5, 4], dtype=ctype))
        OpArgMngr.add_workload('cumsum', np.array([[1, 2, 3, 4], [5, 6, 7, 9], [10, 3, 4, 5]], dtype=ctype), axis=0)
        OpArgMngr.add_workload('cumsum', np.array([[1, 2, 3, 4], [5, 6, 7, 9], [10, 3, 4, 5]], dtype=ctype), axis=1)


def _add_workload_ravel():
    OpArgMngr.add_workload('ravel', np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]))


def _add_workload_dot():
    OpArgMngr.add_workload('dot', np.random.normal(size=(2, 4)), np.random.normal(size=(4, 2)))
    OpArgMngr.add_workload('dot', np.random.normal(size=(4, 2)), np.random.normal(size=(2, 1)))
    OpArgMngr.add_workload('dot', np.random.normal(size=(4, 2)), np.random.normal(size=(2,)))
    OpArgMngr.add_workload('dot', np.random.normal(size=(1, 2)), np.random.normal(size=(2, 4)))
    OpArgMngr.add_workload('dot', np.random.normal(size=(2, 4)), np.random.normal(size=(4,)))
    OpArgMngr.add_workload('dot', np.random.normal(size=(1, 2)), np.random.normal(size=(2, 1)))
    OpArgMngr.add_workload('dot', np.ones((3, 1)), np.array([5.3]))
    OpArgMngr.add_workload('dot', np.array([5.3]), np.ones((1, 3)))
    OpArgMngr.add_workload('dot', np.random.normal(size=(1, 1)), np.random.normal(size=(1, 4)))
    OpArgMngr.add_workload('dot', np.random.normal(size=(4, 1)), np.random.normal(size=(1, 1)))

    dims = [(), (1,), (1, 1)]
    for (dim1, dim2) in itertools.product(dims, dims):
        b1 = np.zeros(dim1)
        b2 = np.zeros(dim2)
        OpArgMngr.add_workload('dot', b1, b2)
    OpArgMngr.add_workload('dot', np.array([[1, 2], [3, 4]], dtype=float), np.array([[1, 0], [1, 1]], dtype=float))
    OpArgMngr.add_workload('dot', np.random.normal(size=(1024, 16)), np.random.normal(size=(16, 32)))


def _add_workload_fix():
    OpArgMngr.add_workload('fix', np.array([[1.0, 1.1, 1.5, 1.8], [-1.0, -1.1, -1.5, -1.8]]))
    OpArgMngr.add_workload('fix', np.array([3.14]))


def _add_workload_flip():
    OpArgMngr.add_workload('flip', np.random.normal(size=(4, 4)), 1)
    OpArgMngr.add_workload('flip', np.array([[0, 1, 2], [3, 4, 5]]), 1)
    OpArgMngr.add_workload('flip', np.random.normal(size=(4, 4)), 0)
    OpArgMngr.add_workload('flip', np.array([[0, 1, 2], [3, 4, 5]]), 0)
    OpArgMngr.add_workload('flip', np.array([[[0, 1], [2, 3]], [[4, 5], [6, 7]]]), 0)
    OpArgMngr.add_workload('flip', np.array([[[0, 1], [2, 3]], [[4, 5], [6, 7]]]), 1)
    OpArgMngr.add_workload('flip', np.array([[[0, 1], [2, 3]], [[4, 5], [6, 7]]]), 2)
    for i in range(4):
        OpArgMngr.add_workload('flip', np.arange(2 * 3 * 4 * 5).reshape(2, 3, 4, 5), i)
    OpArgMngr.add_workload('flip', np.array([[1, 2, 3], [4, 5, 6]]))
    OpArgMngr.add_workload('flip', np.array([[[0, 1], [2, 3]], [[4, 5], [6, 7]]]), ())
    OpArgMngr.add_workload('flip', np.array([[[0, 1], [2, 3]], [[4, 5], [6, 7]]]), (0, 2))
    OpArgMngr.add_workload('flip', np.array([[[0, 1], [2, 3]], [[4, 5], [6, 7]]]), (1, 2))


def _add_workload_flipud():
    OpArgMngr.add_workload('flipud', np.random.normal(size=(4, 4)))
    OpArgMngr.add_workload('flipud', np.array([[0, 1, 2], [3, 4, 5]]))
    OpArgMngr.add_workload('flipud', np.array([[[0, 1], [2, 3]], [[4, 5], [6, 7]]]))


def _add_workload_fliplr():
    OpArgMngr.add_workload('fliplr', np.random.normal(size=(4, 4)))
    OpArgMngr.add_workload('fliplr', np.array([[0, 1, 2], [3, 4, 5]]))
    OpArgMngr.add_workload('fliplr', np.array([[[0, 1], [2, 3]], [[4, 5], [6, 7]]]))


def _add_workload_max(array_pool):
    OpArgMngr.add_workload('max', array_pool['4x1'])


def _add_workload_amax(array_pool):
    a = np.array([3, 4, 5, 10, -3, -5, 6.0])
    b = np.array([[3, 6.0, 9.0],
                  [4, 10.0, 5.0],
                  [8, 3.0, 2.0]])
    c = np.array(1)
    OpArgMngr.add_workload('amax', array_pool['4x1'])
    OpArgMngr.add_workload('amax', a)
    OpArgMngr.add_workload('amax', b, axis=0)
    OpArgMngr.add_workload('amax', b, axis=1)
    OpArgMngr.add_workload('amax', c)
    OpArgMngr.add_workload('amax', c, axis=None)


def _add_workload_min(array_pool):
    OpArgMngr.add_workload('min', array_pool['4x1'])


def _add_workload_amin(array_pool):
    a = np.array([3, 4, 5, 10, -3, -5, 6.0])
    b = np.array([[3, 6.0, 9.0],
                  [4, 10.0, 5.0],
                  [8, 3.0, 2.0]])
    c = np.array(1)
    OpArgMngr.add_workload('amin', array_pool['4x1'])
    OpArgMngr.add_workload('amin', a)
    OpArgMngr.add_workload('amin', b, axis=0)
    OpArgMngr.add_workload('amin', b, axis=1)
    OpArgMngr.add_workload('amin', c)
    OpArgMngr.add_workload('amin', c, axis=None)


def _add_workload_mean(array_pool):
    OpArgMngr.add_workload('mean', array_pool['4x1'])
    OpArgMngr.add_workload('mean', array_pool['4x1'], axis=0, keepdims=True)
    OpArgMngr.add_workload('mean', np.array([[1, 2, 3], [4, 5, 6]]))
    OpArgMngr.add_workload('mean', np.array([]).reshape(2,0,0))
    OpArgMngr.add_workload('mean', np.array([[1, 2, 3], [4, 5, 6]]), axis=0)
    OpArgMngr.add_workload('mean', np.array([[1, 2, 3], [4, 5, 6]]), axis=1)


def _add_workload_ones_like(array_pool):
    OpArgMngr.add_workload('ones_like', array_pool['4x1'])


def _add_workload_atleast_nd():
    a_0 = np.array(1)
    b_0 = np.array(2)
    a_1 = np.array([1, 2])
    b_1 = np.array([2, 3])
    a_2 = np.array([[1, 2], [1, 2]])
    b_2 = np.array([[2, 3], [2, 3]])
    a_3 = [a_2, a_2]
    b_3 = [b_2, b_2]

    OpArgMngr.add_workload('atleast_1d', a_0, b_0)
    OpArgMngr.add_workload('atleast_1d', a_1, b_1)
    OpArgMngr.add_workload('atleast_1d', a_2, b_2)
    OpArgMngr.add_workload('atleast_1d', a_3, b_3)
    OpArgMngr.add_workload('atleast_2d', a_0, b_0)
    OpArgMngr.add_workload('atleast_2d', a_1, b_1)
    OpArgMngr.add_workload('atleast_2d', a_2, b_2)
    OpArgMngr.add_workload('atleast_2d', a_3, b_3)
    OpArgMngr.add_workload('atleast_3d', a_0, b_0)
    OpArgMngr.add_workload('atleast_3d', a_1, b_1)
    OpArgMngr.add_workload('atleast_3d', a_2, b_2)
    OpArgMngr.add_workload('atleast_3d', a_3, b_3)


def _add_workload_prod(array_pool):
    OpArgMngr.add_workload('prod', array_pool['4x1'])
    OpArgMngr.add_workload('prod', np.array([]).reshape(2,0,0))


def _add_workload_product(array_pool):
    OpArgMngr.add_workload('product', array_pool['4x1'])


def _add_workload_repeat(array_pool):
    OpArgMngr.add_workload('repeat', array_pool['4x1'], 3)
    OpArgMngr.add_workload('repeat', np.array(_np.arange(12).reshape(4, 3)[:, 2]), 3)
    m = _np.array([1, 2, 3, 4, 5, 6])
    m_rect = m.reshape((2, 3))

    OpArgMngr.add_workload('repeat', np.array(m), [1, 3, 2, 1, 1, 2]) # Argument "repeats" only supports int
    OpArgMngr.add_workload('repeat', np.array(m), 2)
    B = np.array(m_rect)
    OpArgMngr.add_workload('repeat', B, [2, 1], axis=0)  # Argument "repeats" only supports int
    OpArgMngr.add_workload('repeat', B, [1, 3, 2], axis=1)  # Argument "repeats" only supports int
    OpArgMngr.add_workload('repeat', B, 2, axis=0)
    OpArgMngr.add_workload('repeat', B, 2, axis=1)

    # test_repeat_broadcasting
    a = _np.arange(60).reshape(3, 4, 5)
    for axis in itertools.chain(range(-a.ndim, a.ndim), [None]):
        OpArgMngr.add_workload('repeat', np.array(a), 2, axis=axis)
        OpArgMngr.add_workload('repeat', np.array(a), [2], axis=axis)   # Argument "repeats" only supports int


def _add_workload_reshape():
    arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
    OpArgMngr.add_workload('reshape', arr, (2, 6))
    OpArgMngr.add_workload('reshape', arr, (3, 4))
    # OpArgMngr.add_workload('reshape', arr, (3, 4), order='F') # Items are not equal with order='F'
    OpArgMngr.add_workload('reshape', arr, (3, 4), order='C')
    OpArgMngr.add_workload('reshape', np.array(_np.ones(100)), (100, 1, 1))

    # test_reshape_order
    a = np.array(_np.arange(6))
    # OpArgMngr.add_workload('reshape', a, (2, 3), order='F')  # Items are not equal with order='F'
    a = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    b = a[:, 1]
    # OpArgMngr.add_workload('reshape', b, (2, 2), order='F')  # Items are not equal with order='F'

    a = np.array(_np.ones((0, 2)))
    OpArgMngr.add_workload('reshape', a, (-1, 2))


def _add_workload_rint(array_pool):
    OpArgMngr.add_workload('rint', np.array(4607998452777363968))
    OpArgMngr.add_workload('rint', array_pool['4x1'])


def _add_workload_roll():
    # test_roll1d(self)
    OpArgMngr.add_workload('roll', np.array(_np.arange(10)), 2)

    # test_roll2d(self)
    x2 = np.array(_np.reshape(_np.arange(10), (2, 5)))
    OpArgMngr.add_workload('roll', x2, 1)
    OpArgMngr.add_workload('roll', x2, 1, axis=0)
    OpArgMngr.add_workload('roll', x2, 1, axis=1)
    # # Roll multiple axes at once.
    OpArgMngr.add_workload('roll', x2, 1, axis=(0, 1))
    OpArgMngr.add_workload('roll', x2, (1, 0), axis=(0, 1))
    OpArgMngr.add_workload('roll', x2, (-1, 0), axis=(0, 1))
    OpArgMngr.add_workload('roll', x2, (0, 1), axis=(0, 1))
    OpArgMngr.add_workload('roll', x2, (0, -1), axis=(0, 1))
    OpArgMngr.add_workload('roll', x2, (1, 1), axis=(0, 1))
    OpArgMngr.add_workload('roll', x2, (-1, -1), axis=(0, 1))
    # # Roll the same axis multiple times.
    # OpArgMngr.add_workload('roll', x2, 1, axis=(0, 0)) # Check failed: axes[i - 1] < axes[i] (0 vs. 0) : axes have duplicates [0,0]
    # OpArgMngr.add_workload('roll', x2, 1, axis=(1, 1)) # Check failed: axes[i - 1] < axes[i] (1 vs. 1) : axes have duplicates [1,1]
    # # Roll more than one turn in either direction.
    OpArgMngr.add_workload('roll', x2, 6, axis=1)
    OpArgMngr.add_workload('roll', x2, -4, axis=1)
    # # test_roll_empty
    OpArgMngr.add_workload('roll', np.array([]), 1)


def _add_workload_stack(array_pool):
    OpArgMngr.add_workload('stack', [array_pool['4x1']] * 2)
    OpArgMngr.add_workload('stack', [array_pool['4x1']] * 2, 1)
    OpArgMngr.add_workload('stack', [array_pool['4x1']] * 2, -1)
    OpArgMngr.add_workload('stack', [array_pool['4x1']] * 2, -2)
    OpArgMngr.add_workload('stack', np.random.normal(size=(2, 4, 3)), 2)
    OpArgMngr.add_workload('stack', np.random.normal(size=(2, 4, 3)), -3)
    OpArgMngr.add_workload('stack', np.array([[], [], []]), 1)
    OpArgMngr.add_workload('stack', np.array([[], [], []]))


def _add_workload_sum():
    # OpArgMngr.add_workload('sum', np.ones(101, dtype=bool))
    OpArgMngr.add_workload('sum', np.arange(1, 10).reshape((3, 3)), axis=1, keepdims=True)
    OpArgMngr.add_workload('sum', np.ones(500, dtype=np.float32)/10.)
    OpArgMngr.add_workload('sum', np.ones(500, dtype=np.float64)/10.)
    for dt in (np.float16, np.float32, np.float64):
        for v in (0, 1, 2, 7, 8, 9, 15, 16, 19, 127,
                  128, 1024, 1235):
            d = np.arange(1, v + 1, dtype=dt)
            OpArgMngr.add_workload('sum', d)
        d = np.ones(500, dtype=dt)
        OpArgMngr.add_workload('sum', d[::2])
        OpArgMngr.add_workload('sum', d[1::2])
        OpArgMngr.add_workload('sum', d[::3])
        OpArgMngr.add_workload('sum', d[1::3])
        OpArgMngr.add_workload('sum', d[::-2])
        OpArgMngr.add_workload('sum', d[-1::-2])
        OpArgMngr.add_workload('sum', d[::-3])
        OpArgMngr.add_workload('sum', d[-1::-3])
        d = np.ones((1,), dtype=dt)
        d += d
        OpArgMngr.add_workload('sum', d)
    # OpArgMngr.add_workload('sum', np.array([3]), initial=2)
    # OpArgMngr.add_workload('sum', np.array([0.2]), initial=0.1)


def _add_workload_take():
    OpArgMngr.add_workload('take', np.array([[1, 2], [3, 4]], dtype=int), np.array([], int))
    for mode in ['wrap', 'clip']:
        OpArgMngr.add_workload('take', np.array([[1, 2], [3, 4]], dtype=int), np.array(-1, int), mode=mode)
        OpArgMngr.add_workload('take', np.array([[1, 2], [3, 4]], dtype=int), np.array(4, int), mode=mode)
        OpArgMngr.add_workload('take', np.array([[1, 2], [3, 4]], dtype=int), np.array([-1], int), mode=mode)
        OpArgMngr.add_workload('take', np.array([[1, 2], [3, 4]], dtype=int), np.array([4], int), mode=mode)
    x = (np.random.normal(size=24)*100).reshape((2, 3, 4))
    # OpArgMngr.add_workload('take', x, np.array([-1], int), axis=0)
    OpArgMngr.add_workload('take', x, np.array([-1], int), axis=0, mode='clip')
    OpArgMngr.add_workload('take', x, np.array([2], int), axis=0, mode='clip')
    OpArgMngr.add_workload('take', x, np.array([-1], int), axis=0, mode='wrap')
    OpArgMngr.add_workload('take', x, np.array([2], int), axis=0, mode='wrap')
    OpArgMngr.add_workload('take', x, np.array([3], int), axis=0, mode='wrap')


def _add_workload_unique():
    OpArgMngr.add_workload('unique', np.array([5, 7, 1, 2, 1, 5, 7]*10), True, True, True)
    OpArgMngr.add_workload('unique', np.array([]), True, True, True)
    OpArgMngr.add_workload('unique', np.array([[0, 1, 0], [0, 1, 0]]))
    OpArgMngr.add_workload('unique', np.array([[0, 1, 0], [0, 1, 0]]), axis=0)
    OpArgMngr.add_workload('unique', np.array([[0, 1, 0], [0, 1, 0]]), axis=1)
    # OpArgMngr.add_workload('unique', np.arange(10, dtype=np.uint8).reshape(-1, 2).astype(bool), axis=1)


def _add_workload_delete():
    a = np.arange(5)
    nd_a = np.arange(5).repeat(2).reshape(1, 5, 2)
    lims = [-6, -2, 0, 1, 2, 4, 5]
    steps = [-3, -1, 1, 3]
    for start in lims:
        for stop in lims:
            for step in steps:
                s = slice(start, stop, step)
                OpArgMngr.add_workload('delete', a, s)
                OpArgMngr.add_workload('delete', nd_a, s, axis=1)
    # mxnet.numpy arrays, even 0-sized, have a float32 dtype.  Starting with numpy 1.19, the
    # index array's of delete() must be of integer or boolean type, so we force that below.
    OpArgMngr.add_workload('delete', a, np.array([], dtype='int32'), axis=0)
    OpArgMngr.add_workload('delete', a, 0)
    OpArgMngr.add_workload('delete', a, np.array([], dtype='int32'))
    OpArgMngr.add_workload('delete', a, np.array([0, 1], dtype='int32'))
    OpArgMngr.add_workload('delete', a, slice(1, 2))
    OpArgMngr.add_workload('delete', a, slice(1, -2))
    k = np.arange(10).reshape(2, 5)
    OpArgMngr.add_workload('delete', k, slice(60, None), axis=1)


def _add_workload_var(array_pool):
    OpArgMngr.add_workload('var', array_pool['4x1'])
    OpArgMngr.add_workload('var', np.array([_np.float16(1.)]))
    OpArgMngr.add_workload('var', np.array([1]))
    OpArgMngr.add_workload('var', np.array([1.]))
    OpArgMngr.add_workload('var', np.array([[1, 2, 3], [4, 5, 6]]))
    OpArgMngr.add_workload('var', np.array([[1, 2, 3], [4, 5, 6]]), 0)
    OpArgMngr.add_workload('var', np.array([[1, 2, 3], [4, 5, 6]]), 1)
    OpArgMngr.add_workload('var', np.array([_np.nan]))
    OpArgMngr.add_workload('var', np.array([1, -1, 1, -1]))
    OpArgMngr.add_workload('var', np.array([1,2,3,4], dtype='f8'))


def _add_workload_zeros_like(array_pool):
    OpArgMngr.add_workload('zeros_like', array_pool['4x1'])
    OpArgMngr.add_workload('zeros_like', np.random.uniform(size=(3, 3)).astype(np.float64), dtype=np.int64)
    OpArgMngr.add_workload('zeros_like', np.random.uniform(size=(3, 3)).astype(np.float32), dtype=np.float64)
    OpArgMngr.add_workload('zeros_like', np.random.randint(2, size = (3, 3)), dtype=int)


def _add_workload_full_like(array_pool):
    OpArgMngr.add_workload('full_like', array_pool['4x1'], 1)
    OpArgMngr.add_workload('full_like', np.random.uniform(low=0, high=100, size=(1,3,4), dtype='float64'), 1)
    OpArgMngr.add_workload('full_like', np.random.uniform(low=0, high=100, size=(9,3,1)), 2, dtype=np.int64)
    OpArgMngr.add_workload('full_like', np.random.uniform(low=0, high=100, size=(9,3)), _np.nan)
    OpArgMngr.add_workload('full_like', np.random.uniform(low=0, high=100, size=(2,0)), 0, dtype=np.float32)


def _add_workload_outer():
    OpArgMngr.add_workload('outer', np.ones((5)), np.ones((2)))


def _add_workload_kron():
    OpArgMngr.add_workload('kron', np.ones((5)), np.ones((2)))
    OpArgMngr.add_workload('kron', np.arange(16).reshape((4,4)), np.ones((4,4)))
    OpArgMngr.add_workload('kron', np.ones((2,4)), np.zeros((2,4)))
    OpArgMngr.add_workload('kron', np.ones(()), np.ones(()))


def _add_workload_meshgrid():
    OpArgMngr.add_workload('meshgrid', np.array([1, 2, 3]))
    OpArgMngr.add_workload('meshgrid', np.array([1, 2, 3]), np.array([4, 5, 6, 7]))
    OpArgMngr.add_workload('meshgrid', np.array([1, 2, 3]), np.array([4, 5, 6, 7]), indexing='ij')


def _add_workload_abs():
    OpArgMngr.add_workload('abs', np.random.uniform(size=(11,)).astype(np.float32))
    OpArgMngr.add_workload('abs', np.random.uniform(size=(5,)).astype(np.float64))
    OpArgMngr.add_workload('abs', np.array([_np.inf, -_np.inf, _np.nan]))


def _add_workload_fabs():
    OpArgMngr.add_workload('fabs', np.random.uniform(size=(11,)).astype(np.float32))
    OpArgMngr.add_workload('fabs', np.random.uniform(size=(5,)).astype(np.float64))
    OpArgMngr.add_workload('fabs', np.array([_np.inf, -_np.inf, _np.nan]))


def _add_workload_add(array_pool):
    OpArgMngr.add_workload('add', array_pool['4x1'], array_pool['1x2'])
    OpArgMngr.add_workload('add', array_pool['4x1'], 2)
    OpArgMngr.add_workload('add', 2, array_pool['4x1'])
    OpArgMngr.add_workload('add', array_pool['4x1'], array_pool['1x1x0'])


def _add_workload_arctan2():
    OpArgMngr.add_workload('arctan2', np.array([1, -1, 1]), np.array([1, 1, -1]))
    OpArgMngr.add_workload('arctan2', np.array([np.PZERO, np.NZERO]), np.array([np.NZERO, np.NZERO]))
    OpArgMngr.add_workload('arctan2', np.array([np.PZERO, np.NZERO]), np.array([np.PZERO, np.PZERO]))
    OpArgMngr.add_workload('arctan2', np.array([np.PZERO, np.NZERO]), np.array([-1, -1]))
    OpArgMngr.add_workload('arctan2', np.array([np.PZERO, np.NZERO]), np.array([1, 1]))
    OpArgMngr.add_workload('arctan2', np.array([-1, -1]), np.array([np.PZERO, np.NZERO]))
    OpArgMngr.add_workload('arctan2', np.array([1, 1]), np.array([np.PZERO, np.NZERO]))
    OpArgMngr.add_workload('arctan2', np.array([1, -1, 1, -1]), np.array([-_np.inf, -_np.inf, _np.inf, _np.inf]))
    OpArgMngr.add_workload('arctan2', np.array([_np.inf, -_np.inf]), np.array([1, 1]))
    OpArgMngr.add_workload('arctan2', np.array([_np.inf, -_np.inf]), np.array([-_np.inf, -_np.inf]))
    OpArgMngr.add_workload('arctan2', np.array([_np.inf, -_np.inf]), np.array([_np.inf, _np.inf]))


def _add_workload_copysign():
    OpArgMngr.add_workload('copysign', np.array([1, 0, 0]), np.array([-1, -1, 1]))
    OpArgMngr.add_workload('copysign', np.array([-2, 5, 1, 4, 3], dtype=np.float16), np.array([0, 1, 2, 4, 2], dtype=np.float16))


def _add_workload_degrees():
    OpArgMngr.add_workload('degrees', np.array(np.pi))
    OpArgMngr.add_workload('degrees', np.array(-0.5*np.pi))


def _add_workload_true_divide():
    for dt in [np.float32, np.float64, np.float16]:
        OpArgMngr.add_workload('true_divide', np.array([10, 10, -10, -10], dt), np.array([20, -20, 20, -20], dt))


def _add_workload_inner():
    OpArgMngr.add_workload('inner', np.zeros(shape=(1, 80), dtype=np.float64), np.zeros(shape=(1, 80), dtype=np.float64))
    for dt in [np.float32, np.float64]:
        # OpArgMngr.add_workload('inner', np.array(3, dtype=dt)[()], np.array([1, 2], dtype=dt))
        # OpArgMngr.add_workload('inner', np.array([1, 2], dtype=dt), np.array(3, dtype=dt)[()])
        A = np.array([[1, 2], [3, 4]], dtype=dt)
        B = np.array([[1, 3], [2, 4]], dtype=dt)
        C = np.array([1, 1], dtype=dt)
        OpArgMngr.add_workload('inner', A.T, C)
        OpArgMngr.add_workload('inner', C, A.T)
        OpArgMngr.add_workload('inner', B, C)
        OpArgMngr.add_workload('inner', C, B)
        OpArgMngr.add_workload('inner', A, B)
        OpArgMngr.add_workload('inner', A, A)
        OpArgMngr.add_workload('inner', A, A.copy())
        a = np.arange(5).astype(dt)
        b = a[::-1]
        OpArgMngr.add_workload('inner', b, a)
        a = np.arange(24).reshape(2,3,4).astype(dt)
        b = np.arange(24, 48).reshape(2,3,4).astype(dt)
        OpArgMngr.add_workload('inner', a, b)
        OpArgMngr.add_workload('inner', b, a)


def _add_workload_insert():
    a = np.arange(10)
    OpArgMngr.add_workload('insert', a, 0, np.array([0]))
    OpArgMngr.add_workload('insert', a, np.array([], dtype=np.int64), np.array([]))
    OpArgMngr.add_workload('insert', a, np.array([0, 1], dtype=np.int64), np.array([1, 2]))
    OpArgMngr.add_workload('insert', a, slice(1, 2), np.array([1, 2]))
    OpArgMngr.add_workload('insert', a, slice(1, -2, -1), np.array([]))
    OpArgMngr.add_workload('insert', np.array([0, 1, 2]), np.array([1, 1, 1], dtype=np.int64), np.array([3, 4, 5]))
    OpArgMngr.add_workload('insert', np.array(1), 0, np.array([0]))


def _add_workload_interp():
    xp0 = np.linspace(0, 1, 5)
    fp0 = np.linspace(0, 1, 5)
    x0 = np.linspace(0, 1, 50)
    xp1 = np.array([1, 2, 3, 4])
    fp1 = np.array([1, 2, _np.inf, 4])
    x1 = np.array([1, 2, 2.5, 3, 4])
    xp2 = np.arange(0, 10, 0.0001)
    fp2 = np.sin(xp2)
    xp3 = np.array([190, -190, 350, -350])
    fp3 = np.array([5, 10, 3, 4])
    x3 = np.array([-180, -170, -185, 185, -10, -5, 0, 365])

    OpArgMngr.add_workload('interp', x0, xp0, fp0)
    OpArgMngr.add_workload('interp', x1, xp1, fp1)
    OpArgMngr.add_workload('interp', np.pi, xp2, fp2)
    OpArgMngr.add_workload('interp', x3, xp3, fp3, period=360)
    for size in range(1, 10):
        xp = np.arange(size, dtype=np.float64)
        fp = np.ones(size, dtype=np.float64)
        incpts = np.array([-1, 0, size - 1, size], dtype=np.float64)
        decpts = incpts[::-1]
        OpArgMngr.add_workload('interp', incpts, xp, fp)
        OpArgMngr.add_workload('interp', decpts, xp, fp)
        OpArgMngr.add_workload('interp', incpts, xp, fp, left=0)
        OpArgMngr.add_workload('interp', decpts, xp, fp, left=0)
        OpArgMngr.add_workload('interp', incpts, xp, fp, right=2)
        OpArgMngr.add_workload('interp', decpts, xp, fp, right=2)
        OpArgMngr.add_workload('interp', incpts, xp, fp, left=0, right=2)
        OpArgMngr.add_workload('interp', decpts, xp, fp, left=0, right=2)


def _add_workload_hypot():
    OpArgMngr.add_workload('hypot', np.array(1), np.array(1))
    OpArgMngr.add_workload('hypot', np.array(0), np.array(0))
    OpArgMngr.add_workload('hypot', np.array(_np.nan), np.array(_np.nan))
    OpArgMngr.add_workload('hypot', np.array(_np.nan), np.array(1))
    OpArgMngr.add_workload('hypot', np.array(_np.nan), np.array(_np.inf))
    OpArgMngr.add_workload('hypot', np.array(_np.inf), np.array(_np.nan))
    OpArgMngr.add_workload('hypot', np.array(_np.inf), np.array(0))
    OpArgMngr.add_workload('hypot', np.array(0), np.array(_np.inf))
    OpArgMngr.add_workload('hypot', np.array(_np.inf), np.array(_np.inf))
    OpArgMngr.add_workload('hypot', np.array(_np.inf), np.array(23.0))


def _add_workload_lcm():
    OpArgMngr.add_workload('lcm', np.array([12, 120], dtype=np.int8), np.array([20, 200], dtype=np.int8))
    OpArgMngr.add_workload('lcm', np.array([12, 120], dtype=np.uint8), np.array([20, 200], dtype=np.uint8))
    OpArgMngr.add_workload('lcm', np.array(195225786*2, dtype=np.int32), np.array(195225786*5, dtype=np.int32))


def _add_workload_gcd():
    OpArgMngr.add_workload('gcd', np.array([24, 30], dtype=np.int8), np.array([20, 75], dtype=np.int8))
    OpArgMngr.add_workload('gcd', np.array([24, 30], dtype=np.uint8), np.array([20, 75], dtype=np.uint8))
    OpArgMngr.add_workload('gcd', np.array(195225786*2, dtype=np.int32), np.array(195225786*5, dtype=np.int32))


def _add_workload_bitwise_or():
    OpArgMngr.add_workload('bitwise_or', np.array([False, False, True, True], dtype=np.bool),
                           np.array([False, True, False, True], dtype=np.bool))
    for dtype in [np.int8, np.int32, np.int64]:
        zeros = np.array([0], dtype=dtype)
        ones = np.array([-1], dtype=dtype)
        OpArgMngr.add_workload('bitwise_or', zeros, zeros)
        OpArgMngr.add_workload('bitwise_or', ones, zeros)
        OpArgMngr.add_workload('bitwise_or', zeros, ones)
        OpArgMngr.add_workload('bitwise_or', ones, ones)


def _add_workload_bitwise_and():
    OpArgMngr.add_workload('bitwise_and', np.array([False, False, True, True], dtype=np.bool),
                           np.array([False, True, False, True], dtype=np.bool))
    for dtype in [np.int8, np.int32, np.int64]:
        zeros = np.array([0], dtype=dtype)
        ones = np.array([-1], dtype=dtype)
        OpArgMngr.add_workload('bitwise_and', zeros, zeros)
        OpArgMngr.add_workload('bitwise_and', ones, zeros)
        OpArgMngr.add_workload('bitwise_and', zeros, ones)
        OpArgMngr.add_workload('bitwise_and', ones, ones)


def _add_workload_bitwise_xor():
    OpArgMngr.add_workload('bitwise_xor', np.array([False, False, True, True], dtype=np.bool),
                           np.array([False, True, False, True], dtype=np.bool))
    for dtype in [np.int8, np.int32, np.int64]:
        zeros = np.array([0], dtype=dtype)
        ones = np.array([-1], dtype=dtype)
        OpArgMngr.add_workload('bitwise_xor', zeros, zeros)
        OpArgMngr.add_workload('bitwise_xor', ones, zeros)
        OpArgMngr.add_workload('bitwise_xor', zeros, ones)
        OpArgMngr.add_workload('bitwise_xor', ones, ones)


def _add_workload_bitwise_left_shift():
    for dtype in [np.int8, np.int32, np.int64]:
        twenty = np.array([20], dtype=dtype)
        three = np.array([3], dtype=dtype)
        OpArgMngr.add_workload('bitwise_left_shift', twenty, three)
        OpArgMngr.add_workload('bitwise_left_shift', twenty, three)
        OpArgMngr.add_workload('bitwise_left_shift', twenty, three)
        OpArgMngr.add_workload('bitwise_left_shift', twenty, three)
    OpArgMngr.add_workload('bitwise_left_shift', np.array([9223372036854775807], np.int64), np.array([1], np.int64))
    OpArgMngr.add_workload('bitwise_left_shift', np.array([-9223372036854775808], np.int64), np.array([1], np.int64))


def _add_workload_bitwise_right_shift():
    for dtype in [np.int8, np.int32, np.int64]:
        twenty = np.array([20], dtype=dtype)
        three = np.array([3], dtype=dtype)
        OpArgMngr.add_workload('bitwise_right_shift', twenty, three)
        OpArgMngr.add_workload('bitwise_right_shift', twenty, three)
        OpArgMngr.add_workload('bitwise_right_shift', twenty, three)
        OpArgMngr.add_workload('bitwise_right_shift', twenty, three)
    OpArgMngr.add_workload('bitwise_right_shift', np.array([9223372036854775807], np.int64), np.array([1], np.int64))
    OpArgMngr.add_workload('bitwise_right_shift', np.array([-9223372036854775808], np.int64), np.array([1], np.int64))


def _add_workload_ldexp():
    OpArgMngr.add_workload('ldexp', np.array(2., np.float32), np.array(3, np.int8))
    OpArgMngr.add_workload('ldexp', np.array(2., np.float64), np.array(3, np.int8))
    OpArgMngr.add_workload('ldexp', np.array(2., np.float32), np.array(3, np.int32))
    OpArgMngr.add_workload('ldexp', np.array(2., np.float64), np.array(3, np.int32))
    OpArgMngr.add_workload('ldexp', np.array(2., np.float32), np.array(3, np.int64))
    OpArgMngr.add_workload('ldexp', np.array(2., np.float64), np.array(3, np.int64))
    OpArgMngr.add_workload('ldexp', np.array(2., np.float64), np.array(9223372036854775807, np.int64))
    OpArgMngr.add_workload('ldexp', np.array(2., np.float64), np.array(-9223372036854775808, np.int64))


def _add_workload_logaddexp(array_pool):
    OpArgMngr.add_workload('logaddexp', array_pool['4x1'], array_pool['1x2'])
    OpArgMngr.add_workload('logaddexp', array_pool['4x1'], 2)
    OpArgMngr.add_workload('logaddexp', 2, array_pool['4x1'])
    OpArgMngr.add_workload('logaddexp', array_pool['4x1'], array_pool['1x1x0'])


def _add_workload_subtract(array_pool):
    OpArgMngr.add_workload('subtract', array_pool['4x1'], array_pool['1x2'])
    OpArgMngr.add_workload('subtract', array_pool['4x1'], 2)
    OpArgMngr.add_workload('subtract', 2, array_pool['4x1'])
    OpArgMngr.add_workload('subtract', array_pool['4x1'], array_pool['1x1x0'])


def _add_workload_multiply(array_pool):
    OpArgMngr.add_workload('multiply', array_pool['4x1'], array_pool['1x2'])
    OpArgMngr.add_workload('multiply', array_pool['4x1'], 2)
    OpArgMngr.add_workload('multiply', 2, array_pool['4x1'])
    OpArgMngr.add_workload('multiply', array_pool['4x1'], array_pool['1x1x0'])


def _add_workload_power(array_pool):
    OpArgMngr.add_workload('power', array_pool['4x1'], array_pool['1x2'])
    OpArgMngr.add_workload('power', array_pool['4x1'], 2)
    OpArgMngr.add_workload('power', 2, array_pool['4x1'])
    OpArgMngr.add_workload('power', array_pool['4x1'], array_pool['1x1x0'])
    OpArgMngr.add_workload('power', np.array([1, 2, 3], np.int32), 2.00001)
    OpArgMngr.add_workload('power', np.array([15, 15], np.int64), np.array([15, 15], np.int64))
    OpArgMngr.add_workload('power', 0, np.arange(1, 10))


def _add_workload_mod(array_pool):
    OpArgMngr.add_workload('mod', array_pool['4x1'], array_pool['1x2'])
    OpArgMngr.add_workload('mod', array_pool['4x1'], 2)
    OpArgMngr.add_workload('mod', 2, array_pool['4x1'])
    OpArgMngr.add_workload('mod', array_pool['4x1'], array_pool['1x1x0'])


def _add_workload_fmod(array_pool):
    OpArgMngr.add_workload('fmod', array_pool['4x1'], array_pool['1x2'])
    OpArgMngr.add_workload('fmod', array_pool['4x1'], 2)
    OpArgMngr.add_workload('fmod', 2, array_pool['4x1'])
    OpArgMngr.add_workload('fmod', array_pool['4x1'], array_pool['1x1x0'])


def _add_workload_floor_divide(array_pool):
    OpArgMngr.add_workload('floor_divide', array_pool['4x1'], array_pool['1x2'])
    OpArgMngr.add_workload('floor_divide', array_pool['4x1'], 2)
    OpArgMngr.add_workload('floor_divide', 2, array_pool['4x1'])
    OpArgMngr.add_workload('floor_divide', array_pool['4x1'], array_pool['1x1x0'])
    OpArgMngr.add_workload('floor_divide', np.array([-1, -2, -3], np.float32), 1.9999)
    OpArgMngr.add_workload('floor_divide', np.array([1000, -200, -3], np.int64), 3)
    OpArgMngr.add_workload('floor_divide', np.array([1, -2, -3, 4, -5], np.int32), 2.0001)
    OpArgMngr.add_workload('floor_divide', np.array([1, -50, -0.2, 40000, 0], np.float64), -7)


def _add_workload_remainder():
    # test remainder basic
    OpArgMngr.add_workload('remainder', np.array([0, 1, 2, 4, 2], dtype=np.float16),
                           np.array([-2, 5, 1, 4, 3], dtype=np.float16))

    def _signs(dt):
        if dt in [np.uint8]:
            return (+1,)
        else:
            return (+1, -1)

    for ct in _DTYPES:
        for sg1, sg2 in itertools.product(_signs(ct), _signs(ct)):
            a = np.array(sg1*71, dtype=ct)
            b = np.array(sg2*19, dtype=ct)
            OpArgMngr.add_workload('remainder', a, b)

    # test remainder exact
    nlst = list(range(-127, 0))
    plst = list(range(1, 128))
    dividend = nlst + [0] + plst
    divisor = nlst + plst
    arg = list(itertools.product(dividend, divisor))
    tgt = list(divmod(*t) for t in arg)
    a, b = np.array(arg, dtype=int).T
    # convert exact integer results from Python to float so that
    # signed zero can be used, it is checked.
    for dt in [np.float16, np.float32, np.float64]:
        fa = a.astype(dt)
        fb = b.astype(dt)
        OpArgMngr.add_workload('remainder', fa, fb)

    # test_float_remainder_roundoff
    for ct in _FLOAT_DTYPES:
        for sg1, sg2 in itertools.product((+1, -1), (+1, -1)):
            a = np.array(sg1*78*6e-8, dtype=ct)
            b = np.array(sg2*6e-8, dtype=ct)
            OpArgMngr.add_workload('remainder', a, b)

    # test_float_remainder_corner_cases
    # Check remainder magnitude.
    for ct in _FLOAT_DTYPES:
        b = _np.array(1.0, dtype=ct)
        a = np.array(_np.nextafter(_np.array(0.0, dtype=ct), -b), dtype=ct)
        b = np.array(b, dtype=ct)
        OpArgMngr.add_workload('remainder', a, b)
        OpArgMngr.add_workload('remainder', -a, -b)

    # Check nans, inf
    for ct in [np.float16, np.float32, np.float64]:
        fone = np.array(1.0, dtype=ct)
        fzer = np.array(0.0, dtype=ct)
        finf = np.array(_np.inf, dtype=ct)
        fnan = np.array(_np.nan, dtype=ct)
        # OpArgMngr.add_workload('remainder', fone, fzer)  # failed
        OpArgMngr.add_workload('remainder', fone, fnan)
        OpArgMngr.add_workload('remainder', finf, fone)


def _add_workload_maximum(array_pool):
    OpArgMngr.add_workload('maximum', array_pool['4x1'], array_pool['1x2'])
    OpArgMngr.add_workload('maximum', array_pool['4x1'], 2)
    OpArgMngr.add_workload('maximum', 2, array_pool['4x1'])
    OpArgMngr.add_workload('maximum', array_pool['4x1'], array_pool['1x1x0'])


def _add_workload_fmax(array_pool):
    OpArgMngr.add_workload('fmax', array_pool['4x1'], array_pool['1x2'])
    OpArgMngr.add_workload('fmax', array_pool['4x1'], 2)
    OpArgMngr.add_workload('fmax', 2, array_pool['4x1'])
    OpArgMngr.add_workload('fmax', array_pool['4x1'], array_pool['1x1x0'])


def _add_workload_minimum(array_pool):
    OpArgMngr.add_workload('minimum', array_pool['4x1'], array_pool['1x2'])
    OpArgMngr.add_workload('minimum', array_pool['4x1'], 2)
    OpArgMngr.add_workload('minimum', 2, array_pool['4x1'])
    OpArgMngr.add_workload('minimum', array_pool['4x1'], array_pool['1x1x0'])


def _add_workload_fmin(array_pool):
    OpArgMngr.add_workload('fmin', array_pool['4x1'], array_pool['1x2'])
    OpArgMngr.add_workload('fmin', array_pool['4x1'], 2)
    OpArgMngr.add_workload('fmin', 2, array_pool['4x1'])
    OpArgMngr.add_workload('fmin', array_pool['4x1'], array_pool['1x1x0'])


def _add_workload_negative(array_pool):
    OpArgMngr.add_workload('negative', array_pool['4x1'])


def _add_workload_absolute(array_pool):
    OpArgMngr.add_workload('absolute', array_pool['4x1'])


def _add_workload_sign(array_pool):
    OpArgMngr.add_workload('sign', array_pool['4x1'])
    OpArgMngr.add_workload('sign', np.array([-2, 5, 1, 4, 3], dtype=np.float16))
    OpArgMngr.add_workload('sign', np.array([-.1, 0, .1]))
    # OpArgMngr.add_workload('sign', np.array(_np.array([_np.nan]))) # failed


def _add_workload_exp(array_pool):
    OpArgMngr.add_workload('exp', array_pool['4x1'])


def _add_workload_log(array_pool):
    OpArgMngr.add_workload('log', array_pool['4x1'])


def _add_workload_log2(array_pool):
    OpArgMngr.add_workload('log2', array_pool['4x1'])
    OpArgMngr.add_workload('log2', np.array(2.**65))
    OpArgMngr.add_workload('log2', np.array(_np.inf))
    OpArgMngr.add_workload('log2', np.array(1.))


def _add_workload_log1p():
    OpArgMngr.add_workload('log1p', np.array(-1.))
    OpArgMngr.add_workload('log1p', np.array(_np.inf))
    OpArgMngr.add_workload('log1p', np.array(1e-6))


def _add_workload_log10(array_pool):
    OpArgMngr.add_workload('log10', array_pool['4x1'])


def _add_workload_sqrt():
    OpArgMngr.add_workload('sqrt', np.array([1, np.PZERO, np.NZERO, _np.inf, _np.nan]))


def _add_workload_square():
    OpArgMngr.add_workload('square', np.array([-2, 5, 1, 4, 3], dtype=np.float16))


def _add_workload_cbrt():
    OpArgMngr.add_workload('cbrt', np.array(-2.5**3, dtype=np.float32))
    OpArgMngr.add_workload('cbrt', np.array([1., 2., -3., _np.inf, -_np.inf])**3)
    OpArgMngr.add_workload('cbrt', np.array([_np.inf, -_np.inf, _np.nan]))


def _add_workload_reciprocal():
    for ctype in [np.float16, np.float32, np.float64]:
        OpArgMngr.add_workload('reciprocal', np.array([-2, 5, 1, 4, 3], dtype=ctype))
        OpArgMngr.add_workload('reciprocal', np.array([-2, 0, 1, 0, 3], dtype=ctype))
        OpArgMngr.add_workload('reciprocal', np.array([0], dtype=ctype))


def _add_workload_sin(array_pool):
    OpArgMngr.add_workload('sin', array_pool['4x1'])


def _add_workload_cos(array_pool):
    OpArgMngr.add_workload('cos', array_pool['4x1'])


def _add_workload_tan(array_pool):
    OpArgMngr.add_workload('tan', array_pool['4x1'])


def _add_workload_sinh(array_pool):
    OpArgMngr.add_workload('sinh', array_pool['4x1'])


def _add_workload_cosh(array_pool):
    OpArgMngr.add_workload('cosh', array_pool['4x1'])


def _add_workload_tanh(array_pool):
    OpArgMngr.add_workload('tanh', array_pool['4x1'])


def _add_workload_arcsin(array_pool):
    OpArgMngr.add_workload('arcsin', array_pool['4x1'] - 2)


def _add_workload_arccos(array_pool):
    OpArgMngr.add_workload('arccos', array_pool['4x1'] - 2)


def _add_workload_arctan(array_pool):
    OpArgMngr.add_workload('arctan', array_pool['4x1'])


def _add_workload_arcsinh(array_pool):
    OpArgMngr.add_workload('arcsinh', array_pool['4x1'])


def _add_workload_arccosh(array_pool):
    OpArgMngr.add_workload('arccosh', array_pool['4x1'])


def _add_workload_arctanh(array_pool):
    OpArgMngr.add_workload('arctanh', array_pool['4x1'] - 2)


def _add_workload_ceil(array_pool):
    OpArgMngr.add_workload('ceil', array_pool['4x1'])


def _add_workload_turnc(array_pool):
    OpArgMngr.add_workload('trunc', array_pool['4x1'])


def _add_workload_floor(array_pool):
    OpArgMngr.add_workload('floor', array_pool['4x1'])


def _add_workload_logical_not(array_pool):
    OpArgMngr.add_workload('logical_not', np.ones(10, dtype=np.int32))
    OpArgMngr.add_workload('logical_not', array_pool['4x1'])
    OpArgMngr.add_workload('logical_not', np.array([True, False, True, False], dtype=np.bool))


def _add_workload_bitwise_not():
    OpArgMngr.add_workload('bitwise_not', np.array([True, False, True, False], dtype=np.bool))
    for dtype in [np.int8, np.int32, np.int64]:
        zeros = np.array([0], dtype=dtype)
        ones = np.array([-1], dtype=dtype)
        OpArgMngr.add_workload('bitwise_not', zeros)
        OpArgMngr.add_workload('bitwise_not', ones)


def _add_workload_invert():
    OpArgMngr.add_workload('invert', np.array([True, False, True, False], dtype=np.bool))
    for dtype in [np.int8, np.int32, np.int64]:
        zeros = np.array([0], dtype=dtype)
        ones = np.array([-1], dtype=dtype)
        OpArgMngr.add_workload('invert', zeros)
        OpArgMngr.add_workload('invert', ones)


def _add_workload_vdot():
    OpArgMngr.add_workload('vdot', np.random.normal(size=(2, 4)), np.random.normal(size=(4, 2)))
    OpArgMngr.add_workload('vdot', np.random.normal(size=(2, 4)).astype(np.float64), np.random.normal(size=(2, 4)).astype(np.float64))


def _add_workload_matmul():
    OpArgMngr.add_workload('matmul', np.random.normal(size=(2, 4)), np.random.normal(size=(4, 2)))
    dtype = [np.float32, np.float64]
    def test_shapes():
        dims = [((1, 1), (2, 1, 1)),     # broadcast first argument
                ((2, 1, 1), (1, 1)),     # broadcast second argument
                ((2, 1, 1), (2, 1, 1)),  # matrix stack sizes match
                ]
        for dt, (dm1, dm2) in itertools.product(dtype, dims):
            a = np.ones(dm1, dtype=dt)
            b = np.ones(dm2, dtype=dt)
            OpArgMngr.add_workload('matmul', a, b)
        # vector vector returns scalars.
        for dt in dtype:
            a = np.ones((2,), dtype=dt)
            b = np.ones((2,), dtype=dt)
            OpArgMngr.add_workload('matmul', a, b)

    def test_result_types():
        mat = np.ones((1,1))
        vec = np.ones((1,))
        for dt in dtype:
            m = mat.astype(dt)
            v = vec.astype(dt)
            for arg in [(m, v), (v, m), (m, m)]:
                OpArgMngr.add_workload('matmul', *arg)

    def test_scalar_output():
        vec1 = np.array([2])
        vec2 = np.array([3, 4]).reshape(1, -1)
        for dt in dtype:
            v1 = vec1.astype(dt)
            v2 = vec2.astype(dt)
            OpArgMngr.add_workload('matmul', v1, v2)
            OpArgMngr.add_workload('matmul', v2.T, v1)

    def test_vector_vector_values():
        vec1 = np.array([1, 2])
        vec2 = np.array([3, 4]).reshape(-1, 1)
        for dt in dtype:
            v1 = vec1.astype(dt)
            v2 = vec2.astype(dt)
            OpArgMngr.add_workload('matmul', v1, v2)
            # no broadcast, we must make v1 into a 2d ndarray
            OpArgMngr.add_workload('matmul', v2, v1.reshape(1, -1))

    def test_vector_matrix_values():
        vec = np.array([1, 2])
        mat1 = np.array([[1, 2], [3, 4]])
        mat2 = np.stack([mat1]*2, axis=0)
        for dt in dtype:
            v = vec.astype(dt)
            m1 = mat1.astype(dt)
            m2 = mat2.astype(dt)
            OpArgMngr.add_workload('matmul', v, m1)
            OpArgMngr.add_workload('matmul', v, m2)

    def test_matrix_vector_values():
        vec = np.array([1, 2])
        mat1 = np.array([[1, 2], [3, 4]])
        mat2 = np.stack([mat1]*2, axis=0)
        for dt in dtype:
            v = vec.astype(dt)
            m1 = mat1.astype(dt)
            m2 = mat2.astype(dt)
            OpArgMngr.add_workload('matmul', m1, v)
            OpArgMngr.add_workload('matmul', m2, v)

    def test_matrix_matrix_values():
        mat1 = np.array([[1, 2], [3, 4]])
        mat2 = np.array([[1, 0], [1, 1]])
        mat12 = np.stack([mat1, mat2], axis=0)
        mat21 = np.stack([mat2, mat1], axis=0)
        for dt in dtype:
            m1 = mat1.astype(dt)
            m2 = mat2.astype(dt)
            m12 = mat12.astype(dt)
            m21 = mat21.astype(dt)
            # matrix @ matrix
            OpArgMngr.add_workload('matmul', m1, m2)
            OpArgMngr.add_workload('matmul', m2, m1)
            # stacked @ matrix
            OpArgMngr.add_workload('matmul', m12, m1)
            # matrix @ stacked
            OpArgMngr.add_workload('matmul', m1, m12)
            # stacked @ stacked
            OpArgMngr.add_workload('matmul', m12, m21)

    test_shapes()
    test_result_types()
    test_scalar_output()
    test_vector_vector_values()
    test_vector_matrix_values()
    test_matrix_vector_values()
    test_matrix_matrix_values()


def _add_workload_vstack(array_pool):
    OpArgMngr.add_workload('vstack', (array_pool['4x1'], np.random.uniform(size=(5, 1))))
    OpArgMngr.add_workload('vstack', array_pool['4x1'])
    OpArgMngr.add_workload('vstack', array_pool['1x1x0'])


def _add_workload_column_stack():
    OpArgMngr.add_workload('column_stack', (np.array([1, 2, 3]), np.array([2, 3, 4])))
    OpArgMngr.add_workload('column_stack', (np.array([[1], [2], [3]]), np.array([[2], [3], [4]])))
    OpArgMngr.add_workload('column_stack', [np.array(_np.arange(3)) for _ in range(2)])


def _add_workload_hstack(array_pool):
    OpArgMngr.add_workload('hstack', (np.random.uniform(size=(1, 4)), np.random.uniform(size=(1, 4))))
    OpArgMngr.add_workload('hstack', array_pool['4x1'])
    OpArgMngr.add_workload('hstack', array_pool['1x1x0'])


def _add_workload_dstack(array_pool):
    OpArgMngr.add_workload('dstack', (np.random.uniform(size=(5, 1, 2)), np.random.uniform(size=(5, 1, 3))))
    OpArgMngr.add_workload('dstack', array_pool['4x1'])
    OpArgMngr.add_workload('dstack', array_pool['1x1x0'])


def _add_workload_equal(array_pool):
    # TODO(junwu): fp16 does not work yet with TVM generated ops
    # OpArgMngr.add_workload('equal', np.array([0, 1, 2, 4, 2], dtype=np.float16), np.array([-2, 5, 1, 4, 3], dtype=np.float16))
    OpArgMngr.add_workload('equal', np.array([0, 1, 2, 4, 2], dtype=np.float32), np.array([-2, 5, 1, 4, 3], dtype=np.float32))
    # TODO(junwu): mxnet currently does not have a consistent behavior as NumPy in dealing with _np.nan
    # OpArgMngr.add_workload('equal', np.array([_np.nan]), np.array([_np.nan]))
    OpArgMngr.add_workload('equal', array_pool['4x1'], array_pool['1x2'])


def _add_workload_not_equal(array_pool):
    # TODO(junwu): fp16 does not work yet with TVM generated ops
    # OpArgMngr.add_workload('not_equal', np.array([0, 1, 2, 4, 2], dtype=np.float16), np.array([-2, 5, 1, 4, 3], dtype=np.float16))
    OpArgMngr.add_workload('not_equal', np.array([0, 1, 2, 4, 2], dtype=np.float32), np.array([-2, 5, 1, 4, 3], dtype=np.float32))
    # TODO(junwu): mxnet currently does not have a consistent behavior as NumPy in dealing with _np.nan
    # OpArgMngr.add_workload('not_equal', np.array([_np.nan]), np.array([_np.nan]))
    OpArgMngr.add_workload('not_equal', array_pool['4x1'], array_pool['1x2'])


def _add_workload_greater(array_pool):
    # TODO(junwu): fp16 does not work yet with TVM generated ops
    # OpArgMngr.add_workload('greater', np.array([0, 1, 2, 4, 2], dtype=np.float16), np.array([-2, 5, 1, 4, 3], dtype=np.float16))
    OpArgMngr.add_workload('greater', np.array([0, 1, 2, 4, 2], dtype=np.float32), np.array([-2, 5, 1, 4, 3], dtype=np.float32))
    OpArgMngr.add_workload('greater', array_pool['4x1'], array_pool['1x2'])
    OpArgMngr.add_workload('greater', array_pool['4x1'], 2)
    OpArgMngr.add_workload('greater', 2, array_pool['4x1'])
    # TODO(junwu): mxnet currently does not have a consistent behavior as NumPy in dealing with _np.nan
    # OpArgMngr.add_workload('greater', np.array([_np.nan]), np.array([_np.nan]))


def _add_workload_greater_equal(array_pool):
    # TODO(junwu): fp16 does not work yet with TVM generated ops
    # OpArgMngr.add_workload('greater_equal', np.array([0, 1, 2, 4, 2], dtype=np.float16), np.array([-2, 5, 1, 4, 3], dtype=np.float16))
    OpArgMngr.add_workload('greater_equal', np.array([0, 1, 2, 4, 2], dtype=np.float32), np.array([-2, 5, 1, 4, 3], dtype=np.float32))
    OpArgMngr.add_workload('greater_equal', array_pool['4x1'], array_pool['1x2'])
    OpArgMngr.add_workload('greater_equal', array_pool['4x1'], 2)
    OpArgMngr.add_workload('greater_equal', 2, array_pool['4x1'])
    # TODO(junwu): mxnet currently does not have a consistent behavior as NumPy in dealing with _np.nan
    # OpArgMngr.add_workload('greater_equal', np.array([_np.nan]), np.array([_np.nan]))


def _add_workload_less(array_pool):
    # TODO(junwu): fp16 does not work yet with TVM generated ops
    # OpArgMngr.add_workload('less', np.array([0, 1, 2, 4, 2], dtype=np.float16), np.array([-2, 5, 1, 4, 3], dtype=np.float16))
    OpArgMngr.add_workload('less', np.array([0, 1, 2, 4, 2], dtype=np.float32), np.array([-2, 5, 1, 4, 3], dtype=np.float32))
    OpArgMngr.add_workload('less', array_pool['4x1'], array_pool['1x2'])
    OpArgMngr.add_workload('less', array_pool['4x1'], 2)
    OpArgMngr.add_workload('less', 2, array_pool['4x1'])
    # TODO(junwu): mxnet currently does not have a consistent behavior as NumPy in dealing with _np.nan
    # OpArgMngr.add_workload('less', np.array([_np.nan]), np.array([_np.nan]))


def _add_workload_less_equal(array_pool):
    # TODO(junwu): fp16 does not work yet with TVM generated ops
    # OpArgMngr.add_workload('less_equal', np.array([0, 1, 2, 4, 2], dtype=np.float16), np.array([-2, 5, 1, 4, 3], dtype=np.float16))
    OpArgMngr.add_workload('less_equal', np.array([0, 1, 2, 4, 2], dtype=np.float32), np.array([-2, 5, 1, 4, 3], dtype=np.float32))
    OpArgMngr.add_workload('less_equal', array_pool['4x1'], array_pool['1x2'])
    OpArgMngr.add_workload('less_equal', array_pool['4x1'], 2)
    OpArgMngr.add_workload('less_equal', 2, array_pool['4x1'])
    # TODO(junwu): mxnet currently does not have a consistent behavior as NumPy in dealing with _np.nan
    # OpArgMngr.add_workload('less_equal', np.array([_np.nan]), np.array([_np.nan]))


def _add_workload_logical_and(array_pool):
    OpArgMngr.add_workload('logical_and', np.array([0, 1, 2, 4, 2], dtype=np.float32), np.array([-2, 5, 1, 4, 3], dtype=np.float32))
    OpArgMngr.add_workload('logical_and', np.array([False, False, True, True], dtype=np.bool),
                           np.array([False, True, False, True], dtype=np.bool))

def _add_workload_logical_or(array_pool):
    OpArgMngr.add_workload('logical_or', np.array([0, 1, 2, 4, 2], dtype=np.bool), np.array([-2, 5, 1, 4, 3], dtype=np.bool))
    OpArgMngr.add_workload('logical_or', np.array([False, False, True, True], dtype=np.bool),
                           np.array([False, True, False, True], dtype=np.bool))


def _add_workload_logical_xor(array_pool):
    OpArgMngr.add_workload('logical_xor', np.array([0, 1, 2, 4, 2], dtype=np.float32), np.array([-2, 5, 1, 4, 3], dtype=np.float32))
    OpArgMngr.add_workload('logical_xor', np.array([False, False, True, True], dtype=np.bool),
                           np.array([False, True, False, True], dtype=np.bool))


def _add_workload_where():
    c = np.ones(53).astype(bool)
    d = np.ones_like(c)
    e = np.zeros_like(c)
    OpArgMngr.add_workload('where', c, e, e)
    OpArgMngr.add_workload('where', c, d, e)
    OpArgMngr.add_workload('where', c, d, e[0])
    OpArgMngr.add_workload('where', c, d[0], e)
    # OpArgMngr.add_workload('where', c[::2], d[::2], e[::2])
    # OpArgMngr.add_workload('where', c[1::2], d[1::2], e[1::2])
    # OpArgMngr.add_workload('where', c[::3], d[::3], e[::3])
    # OpArgMngr.add_workload('where', c[1::3], d[1::3], e[1::3])
    # OpArgMngr.add_workload('where', c[::-2], d[::-2], e[::-2])
    # OpArgMngr.add_workload('where', c[::-3], d[::-3], e[::-3])
    # OpArgMngr.add_workload('where', c[1::-3], d[1::-3], e[1::-3])
    c = np.array([True, False])
    a = np.zeros((2, 25))
    b = np.ones((2, 25))
    OpArgMngr.add_workload('where', c.reshape((2, 1)), a, b)
    OpArgMngr.add_workload('where', c, a.T, b.T)


def _add_workload_pad():
    array = _np.array([[1, 2, 3], [1, 2, 3]])
    pad_width = ((5, 5), (5,5))
    array = np.array(array)
    OpArgMngr.add_workload('pad', array, pad_width, mode="constant", constant_values=0)
    OpArgMngr.add_workload('pad', array, pad_width, mode="edge")
    OpArgMngr.add_workload('pad', array, pad_width, mode="symmetric", reflect_type="even")
    OpArgMngr.add_workload('pad', array, pad_width, mode="reflect", reflect_type="even")
    OpArgMngr.add_workload('pad', array, pad_width, mode="maximum")
    OpArgMngr.add_workload('pad', array, pad_width, mode="minimum")


def _add_workload_nonzero():
    OpArgMngr.add_workload('nonzero', np.random.randint(0, 2))
    OpArgMngr.add_workload('nonzero', np.random.randint(0, 2, size=()))
    OpArgMngr.add_workload('nonzero', np.random.randint(0, 2, size=(0, 1, 2)))
    OpArgMngr.add_workload('nonzero', np.random.randint(0, 2, size=(0, 1, 0)))
    OpArgMngr.add_workload('nonzero', np.random.randint(0, 2, size=(2, 3, 4)))
    OpArgMngr.add_workload('nonzero', np.array([False, False, False], dtype=np.bool_))
    OpArgMngr.add_workload('nonzero', np.array([True, False, False], dtype=np.bool_))


def _add_workload_diagflat():
    def get_mat(n):
        data = _np.arange(n)
        data = _np.add.outer(data,data)
        return data

    A = np.array([[1,2],[3,4],[5,6]])
    vals = (100 * np.arange(5)).astype('l')
    vals_c = (100 * np.array(get_mat(5)) + 1).astype('l')
    vals_f = _np.array((100 * get_mat(5) + 1), order='F', dtype='l')
    vals_f = np.array(vals_f)

    OpArgMngr.add_workload('diagflat', A, k=2)
    OpArgMngr.add_workload('diagflat', A, k=1)
    OpArgMngr.add_workload('diagflat', A, k=0)
    OpArgMngr.add_workload('diagflat', A, k=-1)
    OpArgMngr.add_workload('diagflat', A, k=-2)
    OpArgMngr.add_workload('diagflat', A, k=-3)
    OpArgMngr.add_workload('diagflat', vals, k=0)
    OpArgMngr.add_workload('diagflat', vals, k=2)
    OpArgMngr.add_workload('diagflat', vals, k=-2)
    OpArgMngr.add_workload('diagflat', vals_c, k=0)
    OpArgMngr.add_workload('diagflat', vals_c, k=2)
    OpArgMngr.add_workload('diagflat', vals_c, k=-2)
    OpArgMngr.add_workload('diagflat', vals_f, k=0)
    OpArgMngr.add_workload('diagflat', vals_f, k=2)
    OpArgMngr.add_workload('diagflat', vals_f, k=-2)


def _add_workload_shape():
    OpArgMngr.add_workload('shape', np.random.uniform(size=()))
    OpArgMngr.add_workload('shape', np.random.uniform(size=(0, 1)))
    OpArgMngr.add_workload('shape', np.random.uniform(size=(2, 3)))


def _add_workload_diff():
    x = np.array([1, 4, 6, 7, 12])
    OpArgMngr.add_workload('diff', x)
    OpArgMngr.add_workload('diff', x, 2)
    OpArgMngr.add_workload('diff', x, 3)
    OpArgMngr.add_workload('diff', np.array([1.1, 2.2, 3.0, -0.2, -0.1]))
    x = np.zeros((10, 20, 30))
    x[:, 1::2, :] = 1
    OpArgMngr.add_workload('diff', x)
    OpArgMngr.add_workload('diff', x, axis=-1)
    OpArgMngr.add_workload('diff', x, axis=0)
    OpArgMngr.add_workload('diff', x, axis=1)
    OpArgMngr.add_workload('diff', x, axis=-2)
    x = 20 * np.random.uniform(size=(10,20,30))
    OpArgMngr.add_workload('diff', x)
    OpArgMngr.add_workload('diff', x, n=2)
    OpArgMngr.add_workload('diff', x, axis=0)
    OpArgMngr.add_workload('diff', x, n=2, axis=0)
    x = np.array([list(range(3))])
    for n in range(1, 5):
        OpArgMngr.add_workload('diff', x, n=n)


def _add_workload_ediff1d():
    x = np.array([1, 3, 6, 7, 1])
    OpArgMngr.add_workload('ediff1d', x)
    OpArgMngr.add_workload('ediff1d', x, 2, 4)
    OpArgMngr.add_workload('ediff1d', x, x, 3)
    OpArgMngr.add_workload('ediff1d', x, x, x)
    OpArgMngr.add_workload('ediff1d', np.array([1.1, 2.2, 3.0, -0.2, -0.1]))
    x = np.random.randint(5, size=(5, 0, 4))
    OpArgMngr.add_workload('ediff1d', x)
    OpArgMngr.add_workload('ediff1d', x, 2, 4)
    OpArgMngr.add_workload('ediff1d', x, x, 3)
    OpArgMngr.add_workload('ediff1d', x, x, x)

def _add_workload_resize():
    OpArgMngr.add_workload('resize', np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.int32), (5, 1))
    OpArgMngr.add_workload('resize', np.eye(3), 3)
    OpArgMngr.add_workload('resize', np.ones(1), ())
    OpArgMngr.add_workload('resize', np.ones(()), (1,))
    OpArgMngr.add_workload('resize', np.eye(3), (3, 2, 1))
    OpArgMngr.add_workload('resize', np.eye(3), (2, 3, 3))
    OpArgMngr.add_workload('resize', np.ones(10), 15)
    OpArgMngr.add_workload('resize', np.zeros((10, 0)), (0, 10))
    OpArgMngr.add_workload('resize', np.zeros((10, 0)), (0, 100))


def _add_workload_empty_like():
    OpArgMngr.add_workload('empty_like', np.random.uniform(low=0, high=100, size=(1,3,4), dtype='float64'))
    OpArgMngr.add_workload('empty_like', np.random.uniform(low=0, high=100, size=(9,3,1)), np.int32)
    OpArgMngr.add_workload('empty_like', np.random.uniform(low=0, high=100, size=(9,3)), 'float32')
    OpArgMngr.add_workload('empty_like', np.random.uniform(low=0, high=100, size=(9,3,1)), np.bool_)
    OpArgMngr.add_workload('empty_like', np.random.uniform(low=0, high=100, size=(0,3)), np.float32)


def _add_workload_nan_to_num():
    array1 = np.array([[-433, 0, 456, _np.inf], [-1, -_np.inf, 0, 1]])
    array2 = np.array([_np.nan, _np.inf, -_np.inf, -574, 0, 23425, 24234,-5])
    array3 = np.array(-_np.inf)
    OpArgMngr.add_workload('nan_to_num', array1, True, 0, 100, -100)
    OpArgMngr.add_workload('nan_to_num', array1, True, 0.00)
    OpArgMngr.add_workload('nan_to_num', array2, True)
    OpArgMngr.add_workload('nan_to_num', array2, True, -2000, 10000, -10000)
    OpArgMngr.add_workload('nan_to_num', array3, True)


def _add_workload_isnan(array_pool):
    OpArgMngr.add_workload('isnan', array_pool['2x4'])


def _add_workload_isinf(array_pool):
    OpArgMngr.add_workload('isinf', array_pool['2x4'])


def _add_workload_isposinf(array_pool):
    OpArgMngr.add_workload('isposinf', array_pool['2x4'])


def _add_workload_isneginf(array_pool):
    OpArgMngr.add_workload('isneginf', array_pool['2x4'])


def _add_workload_isfinite(array_pool):
    OpArgMngr.add_workload('isfinite', array_pool['2x4'])


def _add_workload_polyval():
    p1 = np.arange(20)
    p2 = np.arange(1)
    x1 = np.arange(20)
    x2 = np.ones((3,3))
    x3 = np.array(2)
    OpArgMngr.add_workload('polyval', p1, x1)
    OpArgMngr.add_workload('polyval', p1, x2)
    OpArgMngr.add_workload('polyval', p1, x3)
    OpArgMngr.add_workload('polyval', p2, x1)
    OpArgMngr.add_workload('polyval', p2, x2)
    OpArgMngr.add_workload('polyval', p2, x3)


def _add_workload_linalg_cond():
    A = np.array([[1., 0, 1], [0, -2., 0], [0, 0, 3.]])
    OpArgMngr.add_workload('linalg.cond', A, _np.inf)
    OpArgMngr.add_workload('linalg.cond', A, -_np.inf)
    OpArgMngr.add_workload('linalg.cond', A, 1)
    OpArgMngr.add_workload('linalg.cond', A, -1)
    OpArgMngr.add_workload('linalg.cond', A, 'fro')


def _add_workload_linalg_matrix_power():
    i = np.array([[0, 1], [-1, 0]])
    OpArgMngr.add_workload('linalg.matrix_power', i, 3)


def _add_workload_linalg_matrix_rank():
    shapes = [
        ((4, 3), ()),
        ((4, 3), (1,)),
        ((4, 3), (2, 3,)),
        ((2, 1, 1), (1,)),
        ((2, 3, 3), (2,)),
        ((2, 3, 1, 1), ()),
        ((2, 3, 4, 4), (1, 3)),
        ((2, 3, 4, 5), (2, 3)),
        ((2, 3, 5, 4), (2, 3)),
    ]
    dtypes = (np.float32, np.float64)
    for dtype in dtypes:
        for a_shape, tol_shape in shapes:
            for tol_is_none in [True, False]:
                a_np = _np.asarray(_np.random.uniform(-10., 10., a_shape))
                a = np.array(a_np, dtype=dtype)
                if tol_is_none:
                    OpArgMngr.add_workload('linalg.matrix_rank', a, None, False)
                else:
                    tol_np = _np.random.uniform(10., 20., tol_shape)
                    tol = np.array(tol_np, dtype=dtype)
                    OpArgMngr.add_workload('linalg.matrix_rank', a, tol, False)


def _add_workload_linalg_multi_dot():
    E = np.ones((4,6,6))
    F = np.ones((6,6))
    OpArgMngr.add_workload('linalg.multi_dot', E)
    OpArgMngr.add_workload('linalg.multi_dot', [F,F])


def _add_workload_heaviside():
    x = np.array([[-30.0, -0.1, 0.0, 0.2], [7.5, _np.nan, _np.inf, -_np.inf]], dtype=np.float64)
    OpArgMngr.add_workload('heaviside', x, 0.5)
    OpArgMngr.add_workload('heaviside', x, 1.0)

    x = x.astype(np.float32)
    OpArgMngr.add_workload('heaviside', x, _np.float32(0.5))
    OpArgMngr.add_workload('heaviside', x, _np.float32(1.0))


def _add_workload_spacing():
    OpArgMngr.add_workload('spacing', _np.float64(1))
    OpArgMngr.add_workload('spacing', _np.float32(1))
    OpArgMngr.add_workload('spacing', _np.inf)
    OpArgMngr.add_workload('spacing', -_np.inf)
    OpArgMngr.add_workload('spacing', _np.float64(1e30))
    OpArgMngr.add_workload('spacing', _np.float32(1e30))


def _add_workload_allclose():
    a = np.random.randn(10)
    b = a + np.random.rand(10) * 1e-6
    c = [1e10,1e-7]
    d = [1.00001e10,1e10,1e-7]
    OpArgMngr.add_workload('allclose', a, b)
    # OpArgMngr.add_workload('allclose', c, d)


def _add_workload_alltrue():
    for i in range(256-7):
        e = np.array([True] * 256, dtype=bool)[7::]
        e[i] = False
        OpArgMngr.add_workload('alltrue', e)
    # big array test for blocked libc loops
    for i in list(range(9, 6000, 507)) + [7764, 90021, -10]:
        e = np.array([True] * 100043, dtype=bool)
        e[i] = False
        OpArgMngr.add_workload('alltrue', e)


def _add_workload_apply_along_axis():
    def double(row):
        return row * 2

    m = np.array([[0, 1], [2, 3]], dtype=np.int32)
    OpArgMngr.add_workload('apply_along_axis', double, 0, m)
    OpArgMngr.add_workload('apply_along_axis', double, 1, m)


def _add_workload_apply_over_axes():
    a = np.arange(24).reshape(2, 3, 4)
    OpArgMngr.add_workload('apply_over_axes', _np.sum, a, [0, 2])


def _add_workload_argpartition():
    # TODO: move more test cases from numpy to here
    OpArgMngr.add_workload('argpartition', np.array([]), 0, kind='introselect')
    OpArgMngr.add_workload('argpartition', np.ones(1), 0, kind='introselect')
    for r in ([2, 1], [1, 2], [1, 1], [3, 2, 1], [1, 2, 3], [2, 1, 3], [2, 3, 1],
              [1, 1, 1], [1, 2, 2], [2, 2, 1], [1, 2, 1]):
        d = np.array(r)
        OpArgMngr.add_workload('argpartition', d, 0, kind='introselect')


def _add_workload_argwhere():
    a = np.arange(6).reshape((2, 3))
    b = np.array([4, 0, 2, 1, 3])
    OpArgMngr.add_workload('argwhere', a>1)
    OpArgMngr.add_workload('argwhere', b)


def _add_workload_array_equal():
    a = np.array([1, 2])
    b = np.array([1, 2, 3])
    c = np.array([3, 4])
    d = np.array([1, 3])
    OpArgMngr.add_workload('array_equal', a, a)
    OpArgMngr.add_workload('array_equal', a, b)
    OpArgMngr.add_workload('array_equal', a, c)
    OpArgMngr.add_workload('array_equal', a, d)


def _add_workload_array_equiv():
    a = np.array([1, 2])
    b = np.array([1, 2, 3])
    c = np.array([3, 4])
    d = np.array([1, 3])
    e = np.array([2])
    f = np.array([[1], [2]])
    g = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    OpArgMngr.add_workload('array_equiv', a, a)
    OpArgMngr.add_workload('array_equiv', a, b)
    OpArgMngr.add_workload('array_equiv', a, c)
    OpArgMngr.add_workload('array_equiv', a, d)
    OpArgMngr.add_workload('array_equiv', a, e)
    OpArgMngr.add_workload('array_equiv', a, f)
    OpArgMngr.add_workload('array_equiv', a, g)


def _add_workload_choose():
    a = np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]], dtype=np.int64)
    choices = np.array([-10, 10])
    OpArgMngr.add_workload('choose', a, choices)


def _add_workload_compress():
    a = np.array([[1, 2], [3, 4], [5, 6]])
    b = np.array([0, 1])
    c = np.array([False, True, True])
    d = np.array([False, True])
    OpArgMngr.add_workload('compress', b, a, axis=0)
    OpArgMngr.add_workload('compress', c, a, axis=0)
    OpArgMngr.add_workload('compress', d, a, axis=1)


def _add_workload_corrcoef():
    a = np.array([0, 1, 0])
    b = np.array([1, 0, 1])
    c = np.array(
        [[0.15391142, 0.18045767, 0.14197213],
         [0.70461506, 0.96474128, 0.27906989],
         [0.9297531, 0.32296769, 0.19267156]])
    OpArgMngr.add_workload('corrcoef', a, b)
    OpArgMngr.add_workload('corrcoef', c)


def _add_workload_correlate():
    x = np.array([1, 2, 3, 4, 5])
    xs = np.arange(1, 20)[::3]
    y = np.array([-1, -2, -3])
    OpArgMngr.add_workload('correlate', x, y)
    OpArgMngr.add_workload('correlate', x, y, 'full')
    OpArgMngr.add_workload('correlate', x, y[:-1], 'full')
    OpArgMngr.add_workload('correlate', x[::-1], y, 'full')
    OpArgMngr.add_workload('correlate', xs, y, 'full')
    OpArgMngr.add_workload('correlate', x, y,"same")


def _add_workload_count_nonzero():
    m = np.array([[0, 1, 7, 0, 0], [3, 0, 0, 2, 19]])
    a = np.array([])
    b = np.eye(3)
    OpArgMngr.add_workload('count_nonzero', m, axis=0)
    OpArgMngr.add_workload('count_nonzero', m, axis=1)
    OpArgMngr.add_workload('count_nonzero', a)
    OpArgMngr.add_workload('count_nonzero', b)


def _add_workload_cov():
    x = np.array(np.random.rand(12))
    y = x.reshape(3, 4)
    OpArgMngr.add_workload('cov', x)
    OpArgMngr.add_workload('cov', x, rowvar=False)
    OpArgMngr.add_workload('cov', x, rowvar=False, bias=True)
    OpArgMngr.add_workload('cov', y)
    OpArgMngr.add_workload('cov', y, y[::-1])
    OpArgMngr.add_workload('cov', y, rowvar=False)
    OpArgMngr.add_workload('cov', y, rowvar=False, bias=True)


def _add_workload_cumprod():
    a = np.array([[1, 2], [3, 5]])
    OpArgMngr.add_workload('cumprod', a)
    OpArgMngr.add_workload('cumprod', a, axis=0)
    OpArgMngr.add_workload('cumprod', a, axis=1)


def _add_workload_digitize():
    a = np.array([1, 2, 3, 4])
    b = np.array([1, 3])
    c = np.array([0, 1, 2, 3, 4])
    e = [1, 3]
    OpArgMngr.add_workload('digitize', a, b)
    OpArgMngr.add_workload('digitize', b, c)
    OpArgMngr.add_workload('digitize', a, e)


def _add_workload_divmod():
    a = [0., 1., 2., 3., 4.]
    OpArgMngr.add_workload('divmod', a, 3)


def _add_workload_extract():
    arr = np.arange(12).reshape((3, 4))
    condition = np.array([[ True, False, False,  True],  # np.mod(arr, 3)==0
                          [False, False,  True, False],
                          [False,  True, False, False]])
    OpArgMngr.add_workload('extract', condition, arr)


def _add_workload_flatnonzero(array_pool):
    x = np.array([-2, -1,  0,  1,  2])
    OpArgMngr.add_workload('flatnonzero', array_pool['4x1'])
    OpArgMngr.add_workload('flatnonzero', array_pool['1x2'])
    OpArgMngr.add_workload('flatnonzero', x)


def _add_workload_float_power():
    x1 = np.array([1, 2, 3, 4, 5, 6])
    x2 = np.array([1.0, 2.0, 3.0, 3.0, 2.0, 1.0])
    x3 = np.array([[1, 2, 3, 3, 2, 1], [1, 2, 3, 3, 2, 1]])
    OpArgMngr.add_workload('float_power', x1, 3)
    OpArgMngr.add_workload('float_power', x1, x2)
    OpArgMngr.add_workload('float_power', x1, x3)


def _add_workload_frexp():
    x = np.arange(9)
    OpArgMngr.add_workload('frexp', x)


def _add_workload_histogram2d():
    x = np.array([0.41702200, 0.72032449, 1.1437481e-4, 0.302332573, 0.146755891])
    y = np.array([0.09233859, 0.18626021, 0.34556073, 0.39676747, 0.53881673])
    xedges = np.linspace(0, 1, 10)
    yedges = np.linspace(0, 1, 10)
    OpArgMngr.add_workload('histogram2d', x, y, (xedges, yedges))
    OpArgMngr.add_workload('histogram2d', x, y, xedges)
    OpArgMngr.add_workload('histogram2d', list(range(10)), list(range(10)))


def _add_workload_histogram_bin_edges():
    a = [1, 2, 3, 4]
    b = [1, 2]
    arr = np.array([0.,  0.,  0.,  1.,  2.,  3.,  3.,  4.,  5.])
    # OpArgMngr.add_workload('histogram_bin_edges', a, b)
    OpArgMngr.add_workload('histogram_bin_edges', arr, bins=30, range=(-0.5, 5))
    OpArgMngr.add_workload('histogram_bin_edges', arr, bins='auto', range=(0, 1))


def _add_workload_histogramdd():
    x = np.array([[-.5, .5, 1.5], [-.5, 1.5, 2.5], [-.5, 2.5, .5],
                      [.5,  .5, 1.5], [.5,  1.5, 2.5], [.5,  2.5, 2.5]])
    ed = [[-2, 0, 2], [0, 1, 2, 3], [0, 1, 2, 3]]
    z = [np.squeeze(y) for y in np.split(x, 3, axis=1)]
    OpArgMngr.add_workload('histogramdd', x, (2, 3, 3), range=[[-1, 1], [0, 3], [0, 3]])
    OpArgMngr.add_workload('histogramdd', x, bins=ed, density=True)
    OpArgMngr.add_workload('histogramdd', x, (2, 3, 4), range=[[-1, 1], [0, 3], [0, 4]],
                                          density=True)
    OpArgMngr.add_workload('histogramdd', z, bins=(4, 3, 2),
                                          range=[[-2, 2], [0, 3], [0, 2]])


def _add_workload_i0():
    a = 0
    b = np.array([2, 3, 4])
    # OpArgMngr.add_workload('i0', a)
    OpArgMngr.add_workload('i0', b)


def _add_workload_in1d():
    test = np.array([0, 1, 2, 5, 0])
    states = [0, 2]
    OpArgMngr.add_workload('in1d', test, states)
    OpArgMngr.add_workload('in1d', test, states, invert=True)


def _add_workload_interp():
    x = np.linspace(0, 1, 5)
    y = np.linspace(0, 1, 5)
    x0 = np.linspace(0, 1, 50)
    x1 = 0
    x2 = .3
    x3 = _np.float32(.3)
    OpArgMngr.add_workload('interp', x0, x, y)
    OpArgMngr.add_workload('interp', x1, x, y)
    OpArgMngr.add_workload('interp', x2, x, y)
    OpArgMngr.add_workload('interp', x3, x, y)
    x = np.array([1, 2, 2.5, 3, 4])
    xp = np.array([1, 2, 3, 4])
    fp = np.array([1, 2, _np.inf, 4])
    OpArgMngr.add_workload('interp', x, xp, fp)


def _add_workload_intersect1d():
    a = np.array([5, 7, 1, 2])
    b = np.array([2, 4, 3, 1, 5])
    c = np.array([[2, 4, 5, 6, 6], [4, 7, 8, 7, 2]])
    d = np.array([[3, 2, 7, 7], [10, 12, 8, 7]])
    OpArgMngr.add_workload('intersect1d', a, b, assume_unique=True)
    OpArgMngr.add_workload('intersect1d', a, b)
    OpArgMngr.add_workload('intersect1d', a, b, assume_unique=True,
                                                return_indices=True)
    OpArgMngr.add_workload('intersect1d', c, d)


def _add_workload_isclose():
    a = np.array([1e10,1e-7])
    b = np.array([1.00001e10,1e-8])
    c = np.array([1.0, _np.nan])
    d = np.array([0.0, 0.0])
    e = np.array([1e-100, 1e-7])
    OpArgMngr.add_workload('isclose', a, b)
    OpArgMngr.add_workload('isclose', c, c)
    OpArgMngr.add_workload('isclose', c, c, equal_nan=True)
    OpArgMngr.add_workload('isclose', d, e, atol=0.0)


def _add_workload_isin():
    element = 2*np.arange(4).reshape((2, 2))
    test_elements = [1, 2, 4, 8]
    test_set = {1, 2, 4, 8}
    OpArgMngr.add_workload('isin', element, test_elements)
    OpArgMngr.add_workload('isin', element, test_elements, invert=True)
    OpArgMngr.add_workload('isin', element, list(test_set))


def _add_workload_ix_():
    a = np.array([0, 1])
    b = np.array([True, True])
    c = np.array([2, 4])
    d = np.array([False, False, True, False, True])
    OpArgMngr.add_workload('ix_', a, c)
    OpArgMngr.add_workload('ix_', b, c)
    OpArgMngr.add_workload('ix_', b, d)


def _add_workload_lexsort():
    a = np.array([1,5,1,4,3,4,4])
    b = np.array([9,4,0,4,0,2,1])
    OpArgMngr.add_workload('lexsort', (a, b))


def _add_workload_min_scalar_type():
    a = 10
    OpArgMngr.add_workload('min_scalar_type', a)


def _add_workload_mirr():
    val = np.array([-4500, -800, 800, 800, 600, 600, 800, 800, 700, 3000])
    OpArgMngr.add_workload('mirr', val, 0.08, 0.055)


def _add_workload_modf():
    a = np.array([0, 3.5])
    b = -0.5
    OpArgMngr.add_workload('modf', a)
    OpArgMngr.add_workload('modf', b)


def _add_workload_msort():
    A = np.array([[0.44567325, 0.79115165, 0.54900530],
                  [0.36844147, 0.37325583, 0.96098397],
                  [0.64864341, 0.52929049, 0.39172155]])
    OpArgMngr.add_workload('msort', A)


def _add_workload_nanargmax():
    a = np.array([[_np.nan, 4], [2, 3]])
    OpArgMngr.add_workload('nanargmax', a)
    OpArgMngr.add_workload('nanargmax', a, axis=0)
    OpArgMngr.add_workload('nanargmax', a, axis=1)


def _add_workload_nanargmin():
    a = np.array([[_np.nan, 4], [2, 3]])
    OpArgMngr.add_workload('nanargmin', a)
    OpArgMngr.add_workload('nanargmin', a, axis=0)
    OpArgMngr.add_workload('nanargmin', a, axis=1)


def _add_workload_nancumprod():
    a = np.array([[1, 2], [3, _np.nan]])
    OpArgMngr.add_workload('nancumprod', a)
    OpArgMngr.add_workload('nancumprod', a, axis=0)
    OpArgMngr.add_workload('nancumprod', a, axis=1)


def _add_workload_nancumsum():
    a = np.array([[1, 2], [3, _np.nan]])
    OpArgMngr.add_workload('nancumsum', a)
    OpArgMngr.add_workload('nancumsum', a, axis=0)
    OpArgMngr.add_workload('nancumsum', a, axis=1)


def _add_workload_nanmax():
    a = np.array([[1, 2], [3, _np.nan]])
    OpArgMngr.add_workload('nanmax', a)
    OpArgMngr.add_workload('nanmax', a, axis=0)
    OpArgMngr.add_workload('nanmax', a, axis=1)


def _add_workload_nanmedian():
    a = np.array([[10.0, _np.nan, 4], [3, 2, 1]])
    OpArgMngr.add_workload('nanmedian', a)
    OpArgMngr.add_workload('nanmedian', a, axis=0)
    OpArgMngr.add_workload('nanmedian', a, axis=1)


def _add_workload_nanmin():
    a = np.array([[1, 2], [3, _np.nan]])
    OpArgMngr.add_workload('nanmin', a)
    OpArgMngr.add_workload('nanmin', a, axis=0)
    OpArgMngr.add_workload('nanmin', a, axis=1)


def _add_workload_nanpercentile():
    a = np.array([[10.0, _np.nan, 4], [3, 2, 1]])
    OpArgMngr.add_workload('nanpercentile', a, 50)
    OpArgMngr.add_workload('nanpercentile', a, 50, axis=0)
    OpArgMngr.add_workload('nanpercentile', a, 50, axis=1)
    OpArgMngr.add_workload('nanpercentile', a, 50, axis=1, keepdims=True)
    OpArgMngr.add_workload('nanpercentile', a, 50, interpolation='lower')
    OpArgMngr.add_workload('nanpercentile', a, 50, interpolation='higher')
    OpArgMngr.add_workload('nanpercentile', a, 50, interpolation='midpoint')
    OpArgMngr.add_workload('nanpercentile', a, 50, interpolation='nearest')


def _add_workload_nanprod():
    a = 1
    b = np.array([1, _np.nan])
    c = np.array([[1, 2], [3, _np.nan]])
    OpArgMngr.add_workload('nanprod', a)
    OpArgMngr.add_workload('nanprod', b)
    OpArgMngr.add_workload('nanprod', c)
    OpArgMngr.add_workload('nanprod', c, axis=0)


def _add_workload_nanquantile():
    a = np.array([[10.0, _np.nan, 4], [3, 2, 1]])
    OpArgMngr.add_workload('nanquantile', a, 0.4)
    OpArgMngr.add_workload('nanquantile', a, 0.4, axis=0)
    OpArgMngr.add_workload('nanquantile', a, 0.4, axis=1)
    OpArgMngr.add_workload('nanquantile', a, 0.4, axis=1, keepdims=True)
    OpArgMngr.add_workload('nanquantile', a, 0.4, interpolation='lower')
    OpArgMngr.add_workload('nanquantile', a, 0.4, interpolation='higher')
    OpArgMngr.add_workload('nanquantile', a, 0.4, interpolation='midpoint')
    OpArgMngr.add_workload('nanquantile', a, 0.4, interpolation='nearest')


def _add_workload_nanstd():
    OpArgMngr.add_workload('nanstd', np.random.uniform(size=(4, 1)))
    A = np.array([[1, 2, 3], [4, _np.nan, 6]])
    OpArgMngr.add_workload('nanstd', A)
    OpArgMngr.add_workload('nanstd', A, 0)
    OpArgMngr.add_workload('nanstd', A, 1)
    OpArgMngr.add_workload('nanstd', np.array([1, -1, 1, -1]))
    OpArgMngr.add_workload('nanstd', np.array([1, -1, 1, -1]), ddof=1)
    OpArgMngr.add_workload('nanstd', np.array([1, -1, 1, -1]), ddof=2)
    OpArgMngr.add_workload('nanstd', np.arange(10), out=np.array(0.))


def _add_workload_nansum():
    a = 1
    b = np.array([1, _np.nan])
    c = np.array([[1, 2], [3, _np.nan]])
    OpArgMngr.add_workload('nansum', a)
    OpArgMngr.add_workload('nansum', b)
    OpArgMngr.add_workload('nansum', c)
    OpArgMngr.add_workload('nansum', c, axis=0)


def _add_workload_nanvar():
    OpArgMngr.add_workload('nanvar', np.random.uniform(size=(4, 1)))
    A = np.array([[1, 2, 3], [4, _np.nan, 6]])
    OpArgMngr.add_workload('nanvar', A)
    OpArgMngr.add_workload('nanvar', A, 0)
    OpArgMngr.add_workload('nanvar', A, 1)
    OpArgMngr.add_workload('nanvar', np.array([1, -1, 1, -1]))
    OpArgMngr.add_workload('nanvar', np.array([1, -1, 1, -1]), ddof=1)
    OpArgMngr.add_workload('nanvar', np.array([1, -1, 1, -1]), ddof=2)
    OpArgMngr.add_workload('nanvar', np.arange(10), out=np.array(0.))


def _add_workload_ndim():
    a = 1
    b = np.array([[1,2,3],[4,5,6]])
    OpArgMngr.add_workload('ndim', a)
    OpArgMngr.add_workload('ndim', b)


def _add_workload_npv():
    rate, cashflows = 0.281, np.array([-100, 39, 59, 55, 20])
    OpArgMngr.add_workload('npv', rate, cashflows)


def _add_workload_partition():
    a = np.array([3, 4, 2, 1])
    OpArgMngr.add_workload('partition', a, 3)
    OpArgMngr.add_workload('partition', a, (2,3))  #


def _add_workload_piecewise():
    a = np.array([0, 0])
    b = np.array([1, 0])
    c = np.array([1])
    x = np.linspace(-2.5, 2.5, 6)
    y = np.array([[ True,  True,  True, False, False, False],
                  [False, False, False,  True,  True,  True]])
    z = np.array([-1, 1])
    OpArgMngr.add_workload('piecewise', a, b, c)
    OpArgMngr.add_workload('piecewise', x, y, z)


def _add_workload_packbits():
    a = np.array([[[1, 0, 1], [0, 1, 0]],
                  [[1, 1, 0], [0, 0, 1]]], dtype = np.int64)
    OpArgMngr.add_workload('packbits', a)
    OpArgMngr.add_workload('packbits', a, axis=-1)
    OpArgMngr.add_workload('packbits', a, bitorder='little')


def _add_workload_pmt():
    OpArgMngr.add_workload('pmt', 0.1 / 12, 1, 60, 55000)


def _add_workload_poly():
    a = np.array([3, -np.sqrt(2), np.sqrt(2)])
    b = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 0]])
    OpArgMngr.add_workload('poly', a)
    OpArgMngr.add_workload('poly', b)


def _add_workload_polyadd():
    a = np.array([1, 2])
    b = np.array([9, 5, 4])
    OpArgMngr.add_workload('polyadd', a, b)


def _add_workload_polydiv():
    x = np.array([3.0, 5.0, 2.0])
    y = np.array([2.0, 1.0])
    OpArgMngr.add_workload('polydiv', x, y)


def _add_workload_polyfit():
    x = np.array([0.0, 1.0, 2.0, 3.0,  4.0,  5.0])
    y = np.array([0.0, 0.8, 0.9, 0.1, -0.8, -1.0])
    OpArgMngr.add_workload('polyfit', x, y, 3)


def _add_workload_polyint():
    a = np.array([1,2,3])
    OpArgMngr.add_workload('polyint', a)
    OpArgMngr.add_workload('polyint', a, m=2)


def _add_workload_polymul():
    a = np.array([1, 2, 3])
    b = np.array([9, 5, 4])
    OpArgMngr.add_workload('polymul', a, b)


def _add_workload_polysub():
    a = np.array([1, 2, 3])
    b = np.array([9, 5, 4])
    OpArgMngr.add_workload('polysub', a, b)


def _add_workload_positive(array_pool):
    OpArgMngr.add_workload('positive', array_pool['4x1'])


def _add_workload_ppmt():
    OpArgMngr.add_workload('ppmt', 0.1 / 12, 1, 60, 55000)


def _add_workload_promote_types():
    OpArgMngr.add_workload('promote_types', np.float16, np.float64)


def _add_workload_ptp():
    x = np.arange(4).reshape((2,2))
    OpArgMngr.add_workload('ptp', x)
    OpArgMngr.add_workload('ptp', x, axis=0)
    OpArgMngr.add_workload('ptp', x, axis=1)
    OpArgMngr.add_workload('ptp', x, keepdims=True)


def _add_workload_pv():
    a = np.array((0.05, 0.04, 0.03))/12
    OpArgMngr.add_workload('pv', 0.05/12, 10*12, -100, 15692.93)
    OpArgMngr.add_workload('pv', a, 10*12, -100, 15692.93)


def _add_workload_rate():
    OpArgMngr.add_workload('rate', 10, 0, -3500, 10000)


def _add_workload_real():
    a = np.array([1, 3, 5])
    b = 2
    OpArgMngr.add_workload('real', a)
    OpArgMngr.add_workload('real', b)


def _add_workload_real_if_close():
    a = np.array([1, 3, 5])
    b = 2
    OpArgMngr.add_workload('real_if_close', a)
    OpArgMngr.add_workload('real_if_close', b)
    # OpArgMngr.add_workload('real_if_close', b, tol=1000)


def _add_workload_result_type():
    OpArgMngr.add_workload('result_type', 3.0, 2)


def _add_workload_rollaxis():
    a = np.ones((3,4,5,6))
    OpArgMngr.add_workload('rollaxis', a, 3, 1)
    OpArgMngr.add_workload('rollaxis', a, 2)
    OpArgMngr.add_workload('rollaxis', a, 1, 4)


def _add_workload_roots():
    a = np.array([1,2,1])
    OpArgMngr.add_workload('roots', a)


def _add_workload_searchsorted():
    a = np.array([1,2,3,4,5])
    b = np.array([-10, 10, 2, 3])
    OpArgMngr.add_workload('searchsorted', a, 3)
    OpArgMngr.add_workload('searchsorted', a, 3, side='right')
    OpArgMngr.add_workload('searchsorted', a, b)


def _add_workload_select():
    x = np.arange(10)
    condlist = np.array([[ True,  True,  True, False, False,
                           False, False, False, False, False],
                         [ False, False, False, False, False,
                           False,  True,  True,  True, True]], dtype=np.bool)
    choicelist = np.array([[ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9],
                           [ 0,  1,  4,  9, 16, 25, 36, 49, 64, 81]])
    OpArgMngr.add_workload('select', condlist, choicelist)


def _add_workload_setdiff1d():
    a = np.array([1, 2, 3, 2, 4, 1])
    b = np.array([3, 4, 5, 6])
    OpArgMngr.add_workload('setdiff1d', a, b)


def _add_workload_setxor1d():
    a = np.array([1, 2, 3, 2, 4])
    b = np.array([2, 3, 5, 7, 5])
    OpArgMngr.add_workload('setxor1d', a, b)


def _add_workload_signbit():
    a = -1.2
    b = np.array([1, -2.3, 2.1])
    OpArgMngr.add_workload('signbit', a)
    OpArgMngr.add_workload('signbit', b)


def _add_workload_size():
    a = np.array([[1,2,3],[4,5,6]])
    OpArgMngr.add_workload('size', a)
    OpArgMngr.add_workload('size', a, 1)
    OpArgMngr.add_workload('size', a, 0)


def _add_workload_take_along_axis():
    a = np.array([[10, 30, 20], [60, 40, 50]])
    ai = np.argsort(a, axis=1)
    OpArgMngr.add_workload('take_along_axis', a, ai, axis=1)


def _add_workload_trapz():
    a = np.array([1,2,3])
    b = np.arange(6).reshape(2, 3)
    x = np.array([4,6,8])
    OpArgMngr.add_workload('trapz', a)
    OpArgMngr.add_workload('trapz', a, x=x)
    OpArgMngr.add_workload('trapz', a, dx=2)
    OpArgMngr.add_workload('trapz', b, axis=1)
    OpArgMngr.add_workload('trapz', b, axis=0)


def _add_workload_tril_indices_from():
    for dt in ['float16', 'float32', 'float64', 'int32', 'int64', 'int8', 'uint8']:
        OpArgMngr.add_workload('tril_indices_from', np.ones((2, 2), dtype=dt))
        arr = np.array([[1, 1, _np.inf],
                        [1, 1, 1],
                        [_np.inf, 1, 1]])
        OpArgMngr.add_workload('tril_indices_from', arr)
        OpArgMngr.add_workload('tril_indices_from', np.zeros((3, 3), dtype=dt))


def _add_workload_trim_zeros():
    a = np.array((0, 0, 0, 1, 2, 3, 0, 2, 1, 0))
    OpArgMngr.add_workload('trim_zeros', a)
    OpArgMngr.add_workload('trim_zeros', a, 'b')


def _add_workload_triu_indices_from():
    a  =np.array([[1,2,3],[4,5,6],[7,8,9],[10,11,12]])
    OpArgMngr.add_workload('triu_indices_from', a, -1)


def _add_workload_union1d():
    a = np.array([5, 4, 7, 1, 2])
    b = np.array([2, 4, 3, 3, 2, 1, 5])
    x = np.array([[0, 1, 2], [3, 4, 5]])
    y = np.array([0, 1, 2, 3, 4])
    OpArgMngr.add_workload('union1d', a, b)
    OpArgMngr.add_workload('union1d', x, y)


def _add_workload_unpackbits():
    a = np.array([[2], [7], [23]], dtype=np.uint8)
    OpArgMngr.add_workload('unpackbits', a)
    OpArgMngr.add_workload('unpackbits', a, axis=1)


def _add_workload_unwrap():
    phase = np.linspace(0, np.pi, num=5)
    phase[3:] += np.pi
    phase_s = np.vstack((phase,phase))
    OpArgMngr.add_workload('unwrap', phase)
    OpArgMngr.add_workload('unwrap', phase_s, axis=1)


def _add_workload_vander():
    x = np.array([1, 2, 3, 5])
    OpArgMngr.add_workload('vander', x, 3)
    OpArgMngr.add_workload('vander', x, 3, increasing=True)


@use_np
def _prepare_workloads():
    array_pool = {
        '4x1': np.random.uniform(size=(4, 1)) + 2,
        '2x4': np.array([[    -433, float('inf'), 456, _np.inf, _np.nan],
                         [-_np.inf, float("nan"),  -1,       0, _np.inf]]),
        '1x2': np.random.uniform(size=(1, 2)) + 2,
        '1x1x0': np.array([[[]]])
    }

    _add_workload_all()
    _add_workload_any()
    _add_workload_sometrue()
    _add_workload_argmin()
    _add_workload_argmax()
    _add_workload_around()
    _add_workload_round()
    _add_workload_round_()
    _add_workload_argsort()
    _add_workload_sort()
    _add_workload_append()
    _add_workload_bincount()
    _add_workload_broadcast_arrays(array_pool)
    _add_workload_broadcast_to()
    _add_workload_clip()
    _add_workload_concatenate(array_pool)
    _add_workload_copy()
    _add_workload_cross()
    _add_workload_cumsum()
    _add_workload_ravel()
    _add_workload_unravel_index()
    _add_workload_diag_indices_from()
    _add_workload_diag()
    _add_workload_diagonal()
    _add_workload_diagflat()
    _add_workload_dot()
    _add_workload_matmul()
    _add_workload_expand_dims()
    _add_workload_fix()
    _add_workload_flip()
    _add_workload_flipud()
    _add_workload_fliplr()
    _add_workload_max(array_pool)
    _add_workload_amax(array_pool)
    _add_workload_min(array_pool)
    _add_workload_amin(array_pool)
    _add_workload_mean(array_pool)
    _add_workload_nonzero()
    _add_workload_ones_like(array_pool)
    _add_workload_atleast_nd()
    _add_workload_prod(array_pool)
    _add_workload_product(array_pool)
    _add_workload_repeat(array_pool)
    _add_workload_reshape()
    _add_workload_rint(array_pool)
    _add_workload_roll()
    _add_workload_split()
    _add_workload_array_split()
    _add_workload_hsplit()
    _add_workload_vsplit()
    _add_workload_dsplit()
    _add_workload_squeeze()
    _add_workload_stack(array_pool)
    _add_workload_std()
    _add_workload_sum()
    _add_workload_swapaxes()
    _add_workload_take()
    _add_workload_tensordot()
    _add_workload_tile()
    _add_workload_transpose()
    _add_workload_unique()
    _add_workload_delete()
    _add_workload_var(array_pool)
    _add_workload_zeros_like(array_pool)
    _add_workload_linalg_norm()
    _add_workload_linalg_cholesky()
    _add_workload_linalg_qr()
    _add_workload_linalg_inv()
    _add_workload_linalg_solve()
    _add_workload_linalg_det()
    _add_workload_linalg_tensorinv()
    _add_workload_linalg_tensorsolve()
    _add_workload_linalg_lstsq()
    _add_workload_linalg_pinv()
    _add_workload_linalg_eigvals()
    _add_workload_linalg_eig()
    _add_workload_linalg_eigvalsh()
    _add_workload_linalg_eigh()
    _add_workload_linalg_slogdet()
    _add_workload_linalg_cond()
    _add_workload_linalg_matrix_power()
    _add_workload_linalg_matrix_rank()
    _add_workload_linalg_multi_dot()
    _add_workload_trace()
    _add_workload_tril()
    _add_workload_triu()
    _add_workload_outer()
    _add_workload_kron()
    _add_workload_meshgrid()
    _add_workload_einsum()
    _add_workload_abs()
    _add_workload_fabs()
    _add_workload_add(array_pool)
    _add_workload_arctan2()
    _add_workload_copysign()
    _add_workload_degrees()
    _add_workload_true_divide()
    _add_workload_inner()
    _add_workload_insert()
    _add_workload_interp()
    _add_workload_hypot()
    _add_workload_lcm()
    _add_workload_gcd()
    _add_workload_bitwise_and()
    _add_workload_bitwise_xor()
    _add_workload_bitwise_or()
    _add_workload_bitwise_left_shift()
    _add_workload_bitwise_right_shift()
    _add_workload_ldexp()
    _add_workload_logaddexp(array_pool)
    _add_workload_subtract(array_pool)
    _add_workload_multiply(array_pool)
    _add_workload_power(array_pool)
    _add_workload_mod(array_pool)
    _add_workload_fmod(array_pool)
    _add_workload_floor_divide(array_pool)
    _add_workload_remainder()
    _add_workload_maximum(array_pool)
    _add_workload_fmax(array_pool)
    _add_workload_minimum(array_pool)
    _add_workload_fmin(array_pool)
    _add_workload_negative(array_pool)
    _add_workload_absolute(array_pool)
    _add_workload_sign(array_pool)
    _add_workload_exp(array_pool)
    _add_workload_log(array_pool)
    _add_workload_log2(array_pool)
    _add_workload_log1p()
    _add_workload_log10(array_pool)
    _add_workload_expm1()
    _add_workload_sqrt()
    _add_workload_square()
    _add_workload_cbrt()
    _add_workload_reciprocal()
    _add_workload_sin(array_pool)
    _add_workload_cos(array_pool)
    _add_workload_tan(array_pool)
    _add_workload_sinh(array_pool)
    _add_workload_cosh(array_pool)
    _add_workload_tanh(array_pool)
    _add_workload_arcsin(array_pool)
    _add_workload_arccos(array_pool)
    _add_workload_arctan(array_pool)
    _add_workload_arcsinh(array_pool)
    _add_workload_arccosh(array_pool)
    _add_workload_arctanh(array_pool)
    _add_workload_ceil(array_pool)
    _add_workload_turnc(array_pool)
    _add_workload_floor(array_pool)
    _add_workload_logical_not(array_pool)
    _add_workload_bitwise_not()
    _add_workload_invert()
    _add_workload_vdot()
    _add_workload_vstack(array_pool)
    _add_workload_column_stack()
    _add_workload_hstack(array_pool)
    _add_workload_dstack(array_pool)
    _add_workload_equal(array_pool)
    _add_workload_not_equal(array_pool)
    _add_workload_greater(array_pool)
    _add_workload_greater_equal(array_pool)
    _add_workload_less(array_pool)
    _add_workload_less_equal(array_pool)
    _add_workload_logical_and(array_pool)
    _add_workload_logical_or(array_pool)
    _add_workload_logical_xor(array_pool)
    _add_workload_where()
    _add_workload_shape()
    _add_workload_diff()
    _add_workload_ediff1d()
    _add_workload_quantile()
    _add_workload_median(array_pool)
    _add_workload_percentile()
    _add_workload_resize()
    _add_workload_full_like(array_pool)
    _add_workload_empty_like()
    _add_workload_nan_to_num()
    _add_workload_polyval()
    _add_workload_isnan(array_pool)
    _add_workload_isinf(array_pool)
    _add_workload_isposinf(array_pool)
    _add_workload_isneginf(array_pool)
    _add_workload_isfinite(array_pool)
    _add_workload_heaviside()
    _add_workload_spacing()
    _add_workload_allclose()
    _add_workload_alltrue()
    _add_workload_apply_along_axis()
    _add_workload_apply_over_axes()
    _add_workload_argpartition()
    _add_workload_argwhere()
    _add_workload_array_equal()
    _add_workload_array_equiv()
    _add_workload_choose()
    _add_workload_compress()
    _add_workload_corrcoef()
    _add_workload_correlate()
    _add_workload_count_nonzero()
    _add_workload_cov()
    _add_workload_cumprod()
    _add_workload_digitize()
    _add_workload_divmod()
    _add_workload_extract()
    _add_workload_flatnonzero(array_pool)
    _add_workload_float_power()
    _add_workload_frexp()
    _add_workload_histogram2d()
    _add_workload_histogram_bin_edges()
    _add_workload_histogramdd()
    _add_workload_i0()
    _add_workload_in1d()
    _add_workload_interp()
    _add_workload_intersect1d()
    _add_workload_isclose()
    _add_workload_isin()
    _add_workload_ix_()
    _add_workload_lexsort()
    _add_workload_min_scalar_type()
    _add_workload_mirr()
    _add_workload_modf()
    _add_workload_msort()
    _add_workload_nanargmax()
    _add_workload_nanargmin()
    _add_workload_nancumprod()
    _add_workload_nancumsum()
    _add_workload_nanmax()
    _add_workload_nanmedian()
    _add_workload_nanmin()
    _add_workload_nanpercentile()
    _add_workload_nanprod()
    _add_workload_nanquantile()
    _add_workload_nanstd()
    _add_workload_nansum()
    _add_workload_nanvar()
    _add_workload_ndim()
    _add_workload_npv()
    _add_workload_packbits()
    _add_workload_pad()
    _add_workload_partition()
    _add_workload_piecewise()
    _add_workload_pmt()
    _add_workload_poly()
    _add_workload_polyadd()
    _add_workload_polydiv()
    _add_workload_polyfit()
    _add_workload_polyint()
    _add_workload_polymul()
    _add_workload_polysub()
    _add_workload_positive(array_pool)
    _add_workload_ppmt()
    _add_workload_promote_types()
    _add_workload_ptp()
    _add_workload_pv()
    _add_workload_rate()
    _add_workload_real()
    _add_workload_real_if_close()
    _add_workload_result_type()
    _add_workload_rollaxis()
    _add_workload_roots()
    _add_workload_searchsorted()
    _add_workload_select()
    _add_workload_setdiff1d()
    _add_workload_setxor1d()
    _add_workload_signbit()
    _add_workload_size()
    _add_workload_take_along_axis()
    _add_workload_trapz()
    _add_workload_tril_indices_from()
    _add_workload_trim_zeros()
    _add_workload_triu_indices_from()
    _add_workload_union1d()
    _add_workload_unpackbits()
    _add_workload_unwrap()
    _add_workload_vander()


def _get_numpy_op_output(onp_op, *args, **kwargs):
    onp_args = [arg.asnumpy() if isinstance(arg, np.ndarray) else arg for arg in args]
    onp_kwargs = {k: v.asnumpy() if isinstance(v, np.ndarray) else v for k, v in kwargs.items()}
    for i, v in enumerate(onp_args):
        if isinstance(v, (list, tuple)):
            new_arrs = [a.asnumpy() if isinstance(a, np.ndarray) else a for a in v]
            onp_args[i] = new_arrs

    return onp_op(*onp_args, **onp_kwargs)


def _check_interoperability_helper(op_name, rel_tol, abs_tol, *args, **kwargs):
    strs = op_name.split('.')
    if len(strs) == 1:
        onp_op = getattr(_np, op_name)
        mxnp_op = getattr(np, op_name)
    elif len(strs) == 2:
        onp_op = getattr(getattr(_np, strs[0]), strs[1])
        mxnp_op = getattr(getattr(np, strs[0]), strs[1])
    else:
        assert False
    if not is_op_runnable():
        return
    out = mxnp_op(*args, **kwargs)
    expected_out = _get_numpy_op_output(onp_op, *args, **kwargs)
    if isinstance(out, (tuple, list)):
        assert type(out) == type(expected_out)
        for arr, expected_arr in zip(out, expected_out):
            if isinstance(arr, np.ndarray):
                assert_almost_equal(arr.asnumpy(), expected_arr, rtol=rel_tol, atol=abs_tol, use_broadcast=False, equal_nan=True)
            else:
                _np.testing.assert_equal(arr, expected_arr)
    elif isinstance(out, np.ndarray):
        assert_almost_equal(out.asnumpy(), expected_out, rtol=rel_tol, atol=abs_tol, use_broadcast=False, equal_nan=True)
    elif isinstance(out, _np.dtype):
        _np.testing.assert_equal(out, expected_out)
    else:
        assert _np.isscalar(out), "{} is not a scalar type".format(str(type(out)))
        if isinstance(out, _np.float):
            _np.testing.assert_almost_equal(out, expected_out)
        else:
            _np.testing.assert_equal(out, expected_out)


def check_interoperability(op_list):
    OpArgMngr.randomize_workloads()
    for name in op_list:
        if name in _TVM_OPS and not is_op_runnable():
            continue
        if name in ['shares_memory', 'may_share_memory', 'empty_like',
                    '__version__', 'dtype', '_NoValue']:  # skip list
            continue
        if name in ['full_like', 'zeros_like', 'ones_like'] and \
                StrictVersion(platform.python_version()) < StrictVersion('3.0.0'):
            continue
        default_tols = (1e-3, 1e-4)
        tols = {'linalg.tensorinv': (1e-2, 5e-3),
                'linalg.solve':     (1e-3, 5e-2)}
        (rel_tol, abs_tol) = tols.get(name, default_tols)
        print('Dispatch test:', name)
        workloads = OpArgMngr.get_workloads(name)
        assert workloads is not None, 'Workloads for operator `{}` has not been ' \
                                      'added for checking interoperability with ' \
                                      'the official NumPy.'.format(name)
        for workload in workloads:
            _check_interoperability_helper(name, rel_tol, abs_tol, *workload['args'], **workload['kwargs'])


@use_np
@with_array_function_protocol
@pytest.mark.serial
def test_np_memory_array_function():
    ops = [_np.shares_memory, _np.may_share_memory]
    for op in ops:
        data_mx = np.zeros([13, 21, 23, 22], dtype=np.float32)
        data_np = _np.zeros([13, 21, 23, 22], dtype=np.float32)
        assert op(data_mx[0,:,:,:], data_mx[1,:,:,:]) == op(data_np[0,:,:,:], data_np[1,:,:,:])
        assert op(data_mx[0,0,0,2:5], data_mx[0,0,0,4:7]) == op(data_np[0,0,0,2:5], data_np[0,0,0,4:7])
        assert op(data_mx, np.ones((5, 0))) == op(data_np, _np.ones((5, 0)))


@use_np
@with_array_function_protocol
@pytest.mark.serial
def test_np_array_function_protocol():
    check_interoperability(_NUMPY_ARRAY_FUNCTION_LIST)


@use_np
@with_array_ufunc_protocol
@pytest.mark.serial
def test_np_array_ufunc_protocol():
    prev_state = util.set_flush_denorms(False)
    try:
        check_interoperability(_NUMPY_ARRAY_UFUNC_LIST)
    finally:
        util.set_flush_denorms(prev_state)


@use_np
@pytest.mark.serial
def test_np_fallback_ops():
    op_list = np.fallback.__all__ + ['linalg.{}'.format(op_name) for op_name in np.fallback_linalg.__all__]
    check_interoperability(op_list)
