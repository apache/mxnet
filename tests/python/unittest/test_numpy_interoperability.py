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
import itertools
import numpy as _np
from mxnet import np
from mxnet.test_utils import assert_almost_equal
from mxnet.test_utils import use_np
from mxnet.test_utils import is_op_runnable
from common import assertRaises, with_seed
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
    'greater_equal'
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
        return OpArgMngr._args.get(name, None)


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
    out = np.empty(4, np.float32)
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
    assertRaises(ValueError, np.split, np.arange(10), 3)


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
            for k, src in enumerate((a, b)):
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
        for r in reps:
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
    for dt in ["double", "float32", "int64"]:
        OpArgMngr.add_workload('linalg.norm', np.array([], dtype=dt))
        OpArgMngr.add_workload('linalg.norm', np.array([np.array([]), np.array([])], dtype=dt))
        # numerical error exceed the tolerance
        if dt == "int64":
            continue
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
        OpArgMngr.add_workload('linalg.norm', A, ord=None,axis=None, keepdims=True)
        for k in range(A.ndim):
            OpArgMngr.add_workload('linalg.norm', A, axis=k)
            OpArgMngr.add_workload('linalg.norm', A, axis=k, keepdims=True)
        for k in itertools.permutations(range(A.ndim), 2):
            OpArgMngr.add_workload('linalg.norm', A, axis=k)
            OpArgMngr.add_workload('linalg.norm', A, axis=k, keepdims=True)
        OpArgMngr.add_workload('linalg.norm', np.array([[]], dtype=dt))
        A = np.array([[1, 3], [5, 7]], dtype=dt)
        OpArgMngr.add_workload('linalg.norm', A)
        OpArgMngr.add_workload('linalg.norm', A, 'fro')
        A = (1 / 10) * np.array([[1, 2, 3], [6, 0, 5], [3, 2, 1]], dtype=dt)
        OpArgMngr.add_workload('linalg.norm', A)
        OpArgMngr.add_workload('linalg.norm', A, 'fro')
    for dt in [np.float16, np.float32, np.float64]:
        OpArgMngr.add_workload('linalg.norm', np.array([[1, 0, 1], [0, 1, 1]], dtype=dt))
        OpArgMngr.add_workload('linalg.norm', np.array([[1, 0, 1], [0, 1, 1]], dtype=dt), 'fro')


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
        arr = np.array([[1, 1, np.inf],
                        [1, 1, 1],
                        [np.inf, 1, 1]])
        OpArgMngr.add_workload('tril', arr)
        OpArgMngr.add_workload('tril', np.zeros((3, 3), dtype=dt))


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
    OpArgMngr.add_workload('expm1', np.array([np.inf]))
    OpArgMngr.add_workload('expm1', np.array([-np.inf]))
    OpArgMngr.add_workload('expm1', np.array([0.]))
    OpArgMngr.add_workload('expm1', np.array([-0.]))


def _add_workload_argmax():
    OpArgMngr.add_workload('argmax', np.random.uniform(size=(4, 5, 6, 7, 8)), 0)
    OpArgMngr.add_workload('argmax', np.random.uniform(size=(4, 5, 6, 7, 8)), 1)
    OpArgMngr.add_workload('argmax', np.random.uniform(size=(4, 5, 6, 7, 8)), 2)
    OpArgMngr.add_workload('argmax', np.random.uniform(size=(4, 5, 6, 7, 8)), 3)
    OpArgMngr.add_workload('argmax', np.random.uniform(size=(4, 5, 6, 7, 8)), 4)
    # OpArgMngr.add_workload('argmax', np.array([0, 1, 2, 3, np.nan]))
    # OpArgMngr.add_workload('argmax', np.array([0, 1, 2, np.nan, 3]))
    # OpArgMngr.add_workload('argmax', np.array([np.nan, 0, 1, 2, 3]))
    # OpArgMngr.add_workload('argmax', np.array([np.nan, 0, np.nan, 2, 3]))
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
    # OpArgMngr.add_workload('argmin', np.array([0, 1, 2, 3, np.nan]))
    # OpArgMngr.add_workload('argmin', np.array([0, 1, 2, np.nan, 3]))
    # OpArgMngr.add_workload('argmin', np.array([np.nan, 0, 1, 2, 3]))
    # OpArgMngr.add_workload('argmin', np.array([np.nan, 0, np.nan, 2, 3]))
    OpArgMngr.add_workload('argmin', np.array([False, False, False, False, True]))
    OpArgMngr.add_workload('argmin', np.array([False, False, False, True, False]))
    OpArgMngr.add_workload('argmin', np.array([True, False, False, False, False]))
    OpArgMngr.add_workload('argmin', np.array([True, False, True, False, False]))


def _add_workload_around():
    OpArgMngr.add_workload('around', np.array([1.56, 72.54, 6.35, 3.25]), decimals=1)


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
    OpArgMngr.add_workload('clip', np.array([-2., np.nan, 0.5, 3., 0.25, np.nan]), -1, 1)


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


def _add_workload_max(array_pool):
    OpArgMngr.add_workload('max', array_pool['4x1'])


def _add_workload_min(array_pool):
    OpArgMngr.add_workload('min', array_pool['4x1'])


def _add_workload_mean(array_pool):
    OpArgMngr.add_workload('mean', array_pool['4x1'])
    OpArgMngr.add_workload('mean', array_pool['4x1'], axis=0, keepdims=True)
    OpArgMngr.add_workload('mean', np.array([[1, 2, 3], [4, 5, 6]]))
    OpArgMngr.add_workload('mean', np.array([[1, 2, 3], [4, 5, 6]]), axis=0)
    OpArgMngr.add_workload('mean', np.array([[1, 2, 3], [4, 5, 6]]), axis=1)


def _add_workload_ones_like(array_pool):
    OpArgMngr.add_workload('ones_like', array_pool['4x1'])


def _add_workload_prod(array_pool):
    OpArgMngr.add_workload('prod', array_pool['4x1'])


def _add_workload_repeat(array_pool):
    OpArgMngr.add_workload('repeat', array_pool['4x1'], 3)
    OpArgMngr.add_workload('repeat', np.array(_np.arange(12).reshape(4, 3)[:, 2]), 3)
    m = _np.array([1, 2, 3, 4, 5, 6])
    m_rect = m.reshape((2, 3))

    # OpArgMngr.add_workload('repeat', np.array(m), [1, 3, 2, 1, 1, 2]) # Argument "repeats" only supports int
    OpArgMngr.add_workload('repeat', np.array(m), 2)
    B = np.array(m_rect)
    # OpArgMngr.add_workload('repeat', B, [2, 1], axis=0)  # Argument "repeats" only supports int
    # OpArgMngr.add_workload('repeat', B, [1, 3, 2], axis=1)  # Argument "repeats" only supports int
    OpArgMngr.add_workload('repeat', B, 2, axis=0)
    OpArgMngr.add_workload('repeat', B, 2, axis=1)

    # test_repeat_broadcasting
    a = _np.arange(60).reshape(3, 4, 5)
    for axis in itertools.chain(range(-a.ndim, a.ndim), [None]):
        OpArgMngr.add_workload('repeat', np.array(a), 2, axis=axis)
    #    OpArgMngr.add_workload('repeat', np.array(a), [2], axis=axis)   # Argument "repeats" only supports int


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


def _add_workload_var(array_pool):
    OpArgMngr.add_workload('var', array_pool['4x1'])
    OpArgMngr.add_workload('var', np.array([np.float16(1.)]))
    OpArgMngr.add_workload('var', np.array([1]))
    OpArgMngr.add_workload('var', np.array([1.]))
    OpArgMngr.add_workload('var', np.array([[1, 2, 3], [4, 5, 6]]))
    OpArgMngr.add_workload('var', np.array([[1, 2, 3], [4, 5, 6]]), 0)
    OpArgMngr.add_workload('var', np.array([[1, 2, 3], [4, 5, 6]]), 1)
    OpArgMngr.add_workload('var', np.array([np.nan]))
    OpArgMngr.add_workload('var', np.array([1, -1, 1, -1]))
    OpArgMngr.add_workload('var', np.array([1,2,3,4], dtype='f8'))


def _add_workload_zeros_like(array_pool):
    OpArgMngr.add_workload('zeros_like', array_pool['4x1'])
    OpArgMngr.add_workload('zeros_like', np.random.uniform(size=(3, 3)).astype(np.float64))
    OpArgMngr.add_workload('zeros_like', np.random.uniform(size=(3, 3)).astype(np.float32))
    OpArgMngr.add_workload('zeros_like', np.random.randint(2, size = (3, 3)))


def _add_workload_outer():
    OpArgMngr.add_workload('outer', np.ones((5)), np.ones((2)))


def _add_workload_meshgrid():
    OpArgMngr.add_workload('meshgrid', np.array([1, 2, 3]))
    OpArgMngr.add_workload('meshgrid', np.array([1, 2, 3]), np.array([4, 5, 6, 7]))
    OpArgMngr.add_workload('meshgrid', np.array([1, 2, 3]), np.array([4, 5, 6, 7]), indexing='ij')


def _add_workload_abs():
    OpArgMngr.add_workload('abs', np.random.uniform(size=(11,)).astype(np.float32))
    OpArgMngr.add_workload('abs', np.random.uniform(size=(5,)).astype(np.float64))
    OpArgMngr.add_workload('abs', np.array([np.inf, -np.inf, np.nan]))


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
    OpArgMngr.add_workload('arctan2', np.array([1, -1, 1, -1]), np.array([-np.inf, -np.inf, np.inf, np.inf]))
    OpArgMngr.add_workload('arctan2', np.array([np.inf, -np.inf]), np.array([1, 1]))
    OpArgMngr.add_workload('arctan2', np.array([np.inf, -np.inf]), np.array([-np.inf, -np.inf]))
    OpArgMngr.add_workload('arctan2', np.array([np.inf, -np.inf]), np.array([np.inf, np.inf]))


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


def _add_workload_hypot():
    OpArgMngr.add_workload('hypot', np.array(1), np.array(1))
    OpArgMngr.add_workload('hypot', np.array(0), np.array(0))
    OpArgMngr.add_workload('hypot', np.array(np.nan), np.array(np.nan))
    OpArgMngr.add_workload('hypot', np.array(np.nan), np.array(1))
    OpArgMngr.add_workload('hypot', np.array(np.nan), np.array(np.inf))
    OpArgMngr.add_workload('hypot', np.array(np.inf), np.array(np.nan))
    OpArgMngr.add_workload('hypot', np.array(np.inf), np.array(0))
    OpArgMngr.add_workload('hypot', np.array(0), np.array(np.inf))
    OpArgMngr.add_workload('hypot', np.array(np.inf), np.array(np.inf))
    OpArgMngr.add_workload('hypot', np.array(np.inf), np.array(23.0))


def _add_workload_lcm():
    OpArgMngr.add_workload('lcm', np.array([12, 120], dtype=np.int8), np.array([20, 200], dtype=np.int8))
    OpArgMngr.add_workload('lcm', np.array([12, 120], dtype=np.uint8), np.array([20, 200], dtype=np.uint8))
    OpArgMngr.add_workload('lcm', np.array(195225786*2, dtype=np.int32), np.array(195225786*5, dtype=np.int32))


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


def _add_workload_ldexp():
    OpArgMngr.add_workload('ldexp', np.array(2., np.float32), np.array(3, np.int8))
    OpArgMngr.add_workload('ldexp', np.array(2., np.float64), np.array(3, np.int8))
    OpArgMngr.add_workload('ldexp', np.array(2., np.float32), np.array(3, np.int32))
    OpArgMngr.add_workload('ldexp', np.array(2., np.float64), np.array(3, np.int32))
    OpArgMngr.add_workload('ldexp', np.array(2., np.float32), np.array(3, np.int64))
    OpArgMngr.add_workload('ldexp', np.array(2., np.float64), np.array(3, np.int64))
    OpArgMngr.add_workload('ldexp', np.array(2., np.float64), np.array(9223372036854775807, np.int64))
    OpArgMngr.add_workload('ldexp', np.array(2., np.float64), np.array(-9223372036854775808, np.int64))


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
        finf = np.array(np.inf, dtype=ct)
        fnan = np.array(np.nan, dtype=ct)
        # OpArgMngr.add_workload('remainder', fone, fzer)  # failed
        OpArgMngr.add_workload('remainder', fone, fnan)
        OpArgMngr.add_workload('remainder', finf, fone)


def _add_workload_maximum(array_pool):
    OpArgMngr.add_workload('maximum', array_pool['4x1'], array_pool['1x2'])
    OpArgMngr.add_workload('maximum', array_pool['4x1'], 2)
    OpArgMngr.add_workload('maximum', 2, array_pool['4x1'])
    OpArgMngr.add_workload('maximum', array_pool['4x1'], array_pool['1x1x0'])


def _add_workload_minimum(array_pool):
    OpArgMngr.add_workload('minimum', array_pool['4x1'], array_pool['1x2'])
    OpArgMngr.add_workload('minimum', array_pool['4x1'], 2)
    OpArgMngr.add_workload('minimum', 2, array_pool['4x1'])
    OpArgMngr.add_workload('minimum', array_pool['4x1'], array_pool['1x1x0'])


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
    OpArgMngr.add_workload('log2', np.array(np.inf))
    OpArgMngr.add_workload('log2', np.array(1.))


def _add_workload_log1p():
    OpArgMngr.add_workload('log1p', np.array(-1.))
    OpArgMngr.add_workload('log1p', np.array(np.inf))
    OpArgMngr.add_workload('log1p', np.array(1e-6))


def _add_workload_log10(array_pool):
    OpArgMngr.add_workload('log10', array_pool['4x1'])


def _add_workload_sqrt():
    OpArgMngr.add_workload('sqrt', np.array([1, np.PZERO, np.NZERO, np.inf, np.nan]))


def _add_workload_square():
    OpArgMngr.add_workload('square', np.array([-2, 5, 1, 4, 3], dtype=np.float16))


def _add_workload_cbrt():
    OpArgMngr.add_workload('cbrt', np.array(-2.5**3, dtype=np.float32))
    OpArgMngr.add_workload('cbrt', np.array([1., 2., -3., np.inf, -np.inf])**3)
    OpArgMngr.add_workload('cbrt', np.array([np.inf, -np.inf, np.nan]))


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


def _add_workload_vdot():
    OpArgMngr.add_workload('vdot', np.random.normal(size=(2, 4)), np.random.normal(size=(4, 2)))
    OpArgMngr.add_workload('vdot', np.random.normal(size=(2, 4)).astype(np.float64), np.random.normal(size=(2, 4)).astype(np.float64))


def _add_workload_vstack(array_pool):
    OpArgMngr.add_workload('vstack', (array_pool['4x1'], np.random.uniform(size=(5, 1))))
    OpArgMngr.add_workload('vstack', array_pool['4x1'])
    OpArgMngr.add_workload('vstack', array_pool['1x1x0'])

def _add_workload_column_stack():
    OpArgMngr.add_workload('column_stack', (np.array([1, 2, 3]), np.array([2, 3, 4])))
    OpArgMngr.add_workload('column_stack', (np.array([[1], [2], [3]]), np.array([[2], [3], [4]])))
    OpArgMngr.add_workload('column_stack', [np.array(_np.arange(3)) for _ in range(2)])


def _add_workload_equal(array_pool):
    # TODO(junwu): fp16 does not work yet with TVM generated ops
    # OpArgMngr.add_workload('equal', np.array([0, 1, 2, 4, 2], dtype=np.float16), np.array([-2, 5, 1, 4, 3], dtype=np.float16))
    OpArgMngr.add_workload('equal', np.array([0, 1, 2, 4, 2], dtype=np.float32), np.array([-2, 5, 1, 4, 3], dtype=np.float32))
    # TODO(junwu): mxnet currently does not have a consistent behavior as NumPy in dealing with np.nan
    # OpArgMngr.add_workload('equal', np.array([np.nan]), np.array([np.nan]))
    OpArgMngr.add_workload('equal', array_pool['4x1'], array_pool['1x2'])


def _add_workload_not_equal(array_pool):
    # TODO(junwu): fp16 does not work yet with TVM generated ops
    # OpArgMngr.add_workload('not_equal', np.array([0, 1, 2, 4, 2], dtype=np.float16), np.array([-2, 5, 1, 4, 3], dtype=np.float16))
    OpArgMngr.add_workload('not_equal', np.array([0, 1, 2, 4, 2], dtype=np.float32), np.array([-2, 5, 1, 4, 3], dtype=np.float32))
    # TODO(junwu): mxnet currently does not have a consistent behavior as NumPy in dealing with np.nan
    # OpArgMngr.add_workload('not_equal', np.array([np.nan]), np.array([np.nan]))
    OpArgMngr.add_workload('not_equal', array_pool['4x1'], array_pool['1x2'])


def _add_workload_greater(array_pool):
    # TODO(junwu): fp16 does not work yet with TVM generated ops
    # OpArgMngr.add_workload('greater', np.array([0, 1, 2, 4, 2], dtype=np.float16), np.array([-2, 5, 1, 4, 3], dtype=np.float16))
    OpArgMngr.add_workload('greater', np.array([0, 1, 2, 4, 2], dtype=np.float32), np.array([-2, 5, 1, 4, 3], dtype=np.float32))
    OpArgMngr.add_workload('greater', array_pool['4x1'], array_pool['1x2'])
    # TODO(junwu): mxnet currently does not have a consistent behavior as NumPy in dealing with np.nan
    # OpArgMngr.add_workload('greater', np.array([np.nan]), np.array([np.nan]))


def _add_workload_greater_equal(array_pool):
    # TODO(junwu): fp16 does not work yet with TVM generated ops
    # OpArgMngr.add_workload('greater_equal', np.array([0, 1, 2, 4, 2], dtype=np.float16), np.array([-2, 5, 1, 4, 3], dtype=np.float16))
    OpArgMngr.add_workload('greater_equal', np.array([0, 1, 2, 4, 2], dtype=np.float32), np.array([-2, 5, 1, 4, 3], dtype=np.float32))
    OpArgMngr.add_workload('greater_equal', array_pool['4x1'], array_pool['1x2'])
    # TODO(junwu): mxnet currently does not have a consistent behavior as NumPy in dealing with np.nan
    # OpArgMngr.add_workload('greater_equal', np.array([np.nan]), np.array([np.nan]))


def _add_workload_less(array_pool):
    # TODO(junwu): fp16 does not work yet with TVM generated ops
    # OpArgMngr.add_workload('less', np.array([0, 1, 2, 4, 2], dtype=np.float16), np.array([-2, 5, 1, 4, 3], dtype=np.float16))
    OpArgMngr.add_workload('less', np.array([0, 1, 2, 4, 2], dtype=np.float32), np.array([-2, 5, 1, 4, 3], dtype=np.float32))
    OpArgMngr.add_workload('less', array_pool['4x1'], array_pool['1x2'])
    # TODO(junwu): mxnet currently does not have a consistent behavior as NumPy in dealing with np.nan
    # OpArgMngr.add_workload('less', np.array([np.nan]), np.array([np.nan]))


def _add_workload_less_equal(array_pool):
    # TODO(junwu): fp16 does not work yet with TVM generated ops
    # OpArgMngr.add_workload('less_equal', np.array([0, 1, 2, 4, 2], dtype=np.float16), np.array([-2, 5, 1, 4, 3], dtype=np.float16))
    OpArgMngr.add_workload('less_equal', np.array([0, 1, 2, 4, 2], dtype=np.float32), np.array([-2, 5, 1, 4, 3], dtype=np.float32))
    OpArgMngr.add_workload('less_equal', array_pool['4x1'], array_pool['1x2'])
    # TODO(junwu): mxnet currently does not have a consistent behavior as NumPy in dealing with np.nan
    # OpArgMngr.add_workload('less_equal', np.array([np.nan]), np.array([np.nan]))


def _add_workload_nonzero():
    OpArgMngr.add_workload('nonzero', np.random.randint(0, 2))
    OpArgMngr.add_workload('nonzero', np.random.randint(0, 2, size=()))
    OpArgMngr.add_workload('nonzero', np.random.randint(0, 2, size=(0, 1, 2)))
    OpArgMngr.add_workload('nonzero', np.random.randint(0, 2, size=(0, 1, 0)))
    OpArgMngr.add_workload('nonzero', np.random.randint(0, 2, size=(2, 3, 4)))
    OpArgMngr.add_workload('nonzero', np.array([False, False, False], dtype=np.bool_))
    OpArgMngr.add_workload('nonzero', np.array([True, False, False], dtype=np.bool_))


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


@use_np
def _prepare_workloads():
    array_pool = {
        '4x1': np.random.uniform(size=(4, 1)) + 2,
        '1x2': np.random.uniform(size=(1, 2)) + 2,
        '1x1x0': np.array([[[]]])
    }

    _add_workload_argmin()
    _add_workload_argmax()
    _add_workload_around()
    _add_workload_append()
    _add_workload_broadcast_arrays(array_pool)
    _add_workload_broadcast_to()
    _add_workload_clip()
    _add_workload_concatenate(array_pool)
    _add_workload_copy()
    _add_workload_cumsum()
    _add_workload_ravel()
    _add_workload_dot()
    _add_workload_expand_dims()
    _add_workload_fix()
    _add_workload_flip()
    _add_workload_max(array_pool)
    _add_workload_min(array_pool)
    _add_workload_mean(array_pool)
    _add_workload_nonzero()
    _add_workload_ones_like(array_pool)
    _add_workload_prod(array_pool)
    _add_workload_repeat(array_pool)
    _add_workload_reshape()
    _add_workload_rint(array_pool)
    _add_workload_roll()
    _add_workload_split()
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
    _add_workload_var(array_pool)
    _add_workload_zeros_like(array_pool)
    _add_workload_linalg_norm()
    _add_workload_trace()
    _add_workload_tril()
    _add_workload_outer()
    _add_workload_meshgrid()
    _add_workload_einsum()
    _add_workload_abs()
    _add_workload_add(array_pool)
    _add_workload_arctan2()
    _add_workload_copysign()
    _add_workload_degrees()
    _add_workload_true_divide()
    _add_workload_inner()
    _add_workload_hypot()
    _add_workload_lcm()
    _add_workload_bitwise_xor()
    _add_workload_ldexp()
    _add_workload_subtract(array_pool)
    _add_workload_multiply(array_pool)
    _add_workload_power(array_pool)
    _add_workload_mod(array_pool)
    _add_workload_remainder()
    _add_workload_maximum(array_pool)
    _add_workload_minimum(array_pool)
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
    _add_workload_vdot()
    _add_workload_vstack(array_pool)
    _add_workload_column_stack()
    _add_workload_equal(array_pool)
    _add_workload_not_equal(array_pool)
    _add_workload_greater(array_pool)
    _add_workload_greater_equal(array_pool)
    _add_workload_less(array_pool)
    _add_workload_less_equal(array_pool)
    _add_workload_shape()
    _add_workload_diff()


_prepare_workloads()


def _get_numpy_op_output(onp_op, *args, **kwargs):
    onp_args = [arg.asnumpy() if isinstance(arg, np.ndarray) else arg for arg in args]
    onp_kwargs = {k: v.asnumpy() if isinstance(v, np.ndarray) else v for k, v in kwargs.items()}
    for i, v in enumerate(onp_args):
        if isinstance(v, (list, tuple)):
            new_arrs = [a.asnumpy() if isinstance(a, np.ndarray) else a for a in v]
            onp_args[i] = new_arrs

    return onp_op(*onp_args, **onp_kwargs)


def _check_interoperability_helper(op_name, *args, **kwargs):
    strs = op_name.split('.')
    if len(strs) == 1:
        onp_op = getattr(_np, op_name)
    elif len(strs) == 2:
        onp_op = getattr(getattr(_np, strs[0]), strs[1])
    else:
        assert False
    if not is_op_runnable():
        return
    out = onp_op(*args, **kwargs)
    expected_out = _get_numpy_op_output(onp_op, *args, **kwargs)
    if isinstance(out, (tuple, list)):
        assert type(out) == type(expected_out)
        for arr, expected_arr in zip(out, expected_out):
            if isinstance(arr, np.ndarray):
                assert_almost_equal(arr.asnumpy(), expected_arr, rtol=1e-3, atol=1e-4, use_broadcast=False, equal_nan=True)
            else:
                _np.testing.assert_equal(arr, expected_arr)
    else:
        assert isinstance(out, np.ndarray)
        assert_almost_equal(out.asnumpy(), expected_out, rtol=1e-3, atol=1e-4, use_broadcast=False, equal_nan=True)


def check_interoperability(op_list):
    for name in op_list:
        if name in _TVM_OPS and not is_op_runnable():
            continue
        if name in ['shares_memory', 'may_share_memory']:  # skip list
            continue
        print('Dispatch test:', name)
        workloads = OpArgMngr.get_workloads(name)
        assert workloads is not None, 'Workloads for operator `{}` has not been ' \
                                      'added for checking interoperability with ' \
                                      'the official NumPy.'.format(name)
        for workload in workloads:
            _check_interoperability_helper(name, *workload['args'], **workload['kwargs'])


@with_seed()
@use_np
@with_array_function_protocol
def test_np_memory_array_function():
    ops = [_np.shares_memory, _np.may_share_memory]
    for op in ops:
        data_mx = np.zeros([13, 21, 23, 22], dtype=np.float32)
        data_np = _np.zeros([13, 21, 23, 22], dtype=np.float32)
        assert op(data_mx[0,:,:,:], data_mx[1,:,:,:]) == op(data_np[0,:,:,:], data_np[1,:,:,:])
        assert op(data_mx[0,0,0,2:5], data_mx[0,0,0,4:7]) == op(data_np[0,0,0,2:5], data_np[0,0,0,4:7])
        assert op(data_mx, np.ones((5, 0))) == op(data_np, _np.ones((5, 0)))


@with_seed()
@use_np
@with_array_function_protocol
def test_np_array_function_protocol():
    check_interoperability(_NUMPY_ARRAY_FUNCTION_LIST)


@with_seed()
@use_np
@with_array_ufunc_protocol
def test_np_array_ufunc_protocol():
    check_interoperability(_NUMPY_ARRAY_UFUNC_LIST)


if __name__ == '__main__':
    import nose
    nose.runmodule()
