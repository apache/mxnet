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

import sys
import numpy as onp
import pytest
import mxnet as mx
from mxnet import np
from mxnet.gluon import HybridBlock
from mxnet.test_utils import assert_almost_equal, use_np, set_default_context, environment
import os
curr_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
sys.path.insert(0, os.path.join(curr_path, '../unittest'))
from common import assertRaises

set_default_context(mx.gpu(0))

@use_np
def test_np_einsum():
    class TestEinsum(HybridBlock):
        def __init__(self, subscripts, optimize):
            super(TestEinsum, self).__init__()
            self.subscripts = subscripts
            self.optimize = optimize

        def forward(self, *operands):
            return mx.np.einsum(self.subscripts, *operands, optimize=self.optimize)

    def dbg(name, data):
        print('type of {} = {}'.format(name, type(data)))
        print('shape of {} = {}'.format(name, data.shape))
        print('{} = {}'.format(name, data))

    configs = [
        ('ii', [(5, 5)], lambda *args: (onp.eye(5),)),
        ('ii->i', [(5, 5)], lambda *args: (onp.eye(5),)),
        ('ij->i', [(5, 5)], lambda *args: (onp.ones((5, 5)),)),
        ('...j->...', [(5, 5)], lambda *args: (onp.ones((5, 5)),)),
        ('ji', [(2, 3)], lambda *args: (onp.ones((2, 3)),)),
        ('ij->ji', [(2, 3)], lambda *args: (onp.ones((2, 3)),)),
        ('ij, jk', [(5, 0), (0, 4)], lambda *args: (onp.empty((5, 0)), onp.empty((0, 4)))),

        ('i, i', [(5,), (5,)], lambda *args: (args[1], args[0])),
        ('ij, j', [(5, 5), (5,)], lambda *args: (onp.tile(args[1][None, :], [5, 1]),
                                                 args[0].sum(axis=0))),
        ('...j, j', [(5, 5), (5,)], lambda *args: (onp.tile(args[1][None, :], [5, 1]),
                                                   onp.sum(args[0], axis=0))),
        ('..., ...', [(), (2, 3)], lambda *args: (onp.sum(args[1], axis=None),
                                                  args[0] * onp.ones((2, 3)))),
        (', ij', [(), (2, 3)], lambda *args: (onp.sum(args[1], axis=None),
                                              args[0] * onp.ones((2, 3)))),
        ('i, j', [(2,), (5, )], lambda *args: (onp.sum(args[1], axis=None) * onp.ones(2),
                                               onp.sum(args[0], axis=None) * onp.ones(5))),
        ('ijk, jil->kl', [(3, 4, 5), (4, 3, 2)], lambda *args: (onp.tile(onp.transpose(onp.sum(args[1],
                                                                  axis=-1))[:, :, None], [1, 1, 5]),
                                                                onp.tile(onp.transpose(onp.sum(args[0],
                                                                  axis=-1))[:, :, None], [1, 1, 2]))),
        ('ijk, jil->kl', [(33, 44, 55), (44, 33, 22)], lambda *args: (onp.tile(onp.transpose(onp.sum(args[1],
                                                                  axis=-1))[:, :, None], [1, 1, 55]),
                                                                onp.tile(onp.transpose(onp.sum(args[0],
                                                                  axis=-1))[:, :, None], [1, 1, 22]))),
        ('ki, jk->ij', [(3, 2), (4, 3)], lambda *args: (onp.tile(args[1].sum(axis=0)[:, None], [1, 2]),
                                                        onp.tile(args[0].sum(axis=1)[None, :], [4, 1]))),
        ('ki, ...k->i...', [(3, 2), (4, 3)], lambda *args: (onp.tile(args[1].sum(axis=0)[:, None], [1, 2]),
                                                            onp.tile(args[0].sum(axis=1)[None, :], [4, 1]))),
        ('k..., jk', [(3, 2), (4, 3)], lambda *args: (onp.tile(args[1].sum(axis=0)[:, None], [1, 2]),
                                                      onp.tile(args[0].sum(axis=1)[None, :], [4, 1]))),
        (('ij,jk'), [(2, 5), (5, 2)],
            lambda *args: (onp.dot(onp.ones((2, 2)), args[1].T),
            onp.dot(args[0].T, onp.ones((2, 2))))),
        (('ij,jk,kl'), [(2, 2), (2, 5), (5, 2)],
            lambda *args: (onp.dot(onp.ones((2, 2)), onp.dot(args[1], args[2]).T),
            onp.dot(args[0].T, onp.dot(onp.ones((2, 2)), args[2].T)),
            onp.dot(onp.dot(args[0], args[1]).T, onp.ones((2, 2))))),
        (('ij,jk,kl->il'), [(2, 2), (2, 5), (5, 2)],
            lambda *args: (onp.dot(onp.ones((2, 2)), onp.dot(args[1], args[2]).T),
            onp.dot(args[0].T, onp.dot(onp.ones((2, 2)), args[2].T)),
            onp.dot(onp.dot(args[0], args[1]).T, onp.ones((2, 2))))),
        (('ij,jk,kl->il'), [(67, 89), (89, 55), (55, 99)],
            lambda *args: (onp.dot(onp.ones((67, 99)), onp.dot(args[1], args[2]).T),
            onp.dot(args[0].T, onp.dot(onp.ones((67, 99)), args[2].T)),
            onp.dot(onp.dot(args[0], args[1]).T, onp.ones((67, 99))))),
        (('ij,jk,kl, lm->im'), [(12, 54), (54, 32), (32, 45), (45, 67)],
            lambda *args: (onp.dot(onp.ones((12, 67)), onp.dot(args[1], onp.dot(args[2], args[3])).T),
            onp.dot(args[0].T, onp.dot(onp.ones((12, 67)), onp.dot(args[2], args[3]).T)),
            onp.dot(onp.dot(args[0], args[1]).T, onp.dot(onp.ones((12, 67)), args[3].T)),
            onp.dot(onp.dot(args[0], onp.dot(args[1], args[2])).T, onp.ones((12, 67))))),

        # broadcast axis
        ('ij, ij -> i', [(1, 4), (2, 4)], lambda *args: (onp.sum(args[1], axis=0)[None, :],
                                                         onp.tile(args[0], [2, 1]))),
        ('...ij, ...jk -> ...ik', [(1, 4), (4, 2)], lambda *args: (args[1].sum(axis=1)[None, :],
                                                                   onp.tile(args[0].sum(axis=0)[: ,None], [1, 2]))),
        ('...ij, ...jk -> ...ik', [(2, 4), (4, 2)], lambda *args: (onp.tile(args[1].sum(axis=1)[None, :], [2, 1]),
                                                                   onp.tile(args[0].sum(axis=0)[: ,None], [1, 2]))),
        ('...ij, ...jk -> ...ik', [(3, 2, 1, 4), (3, 2, 4, 2)], lambda *args: (
                                                            args[1].sum(axis=3)[:, :, None, :],
                                                            onp.tile(args[0].sum(axis=2)[:, :, :, None], [1, 1, 1, 2]))),
        ('...ij, ...ik -> ...jk', [(1, 1, 1, 4), (1, 1, 1, 3)], lambda *args: (
                                                            onp.tile(args[1].sum(axis=3)[:, :, :, None], [1, 1, 1, 4]),
                                                            onp.tile(args[0].sum(axis=3)[:, :, : ,None], [1, 1, 1, 3]))),
        ('...ij, ...jc -> ...ic', [(1, 1, 5, 3), (1, 1, 3, 2)], lambda *args: (
                                                            onp.tile(args[1].sum(axis=3)[:, :, None, :], [1, 1, 5, 1]),
                                                            onp.tile(args[0].sum(axis=2)[:, :, : ,None], [1, 1, 1, 2]))),
        ('...ij, ...jc -> ...ic', [(1, 2, 5, 4), (1, 2, 4, 2)], lambda *args: (
                                                            onp.tile(args[1].sum(axis=3)[:, :, None, :], [1, 1, 5, 1]),
                                                            onp.tile(args[0].sum(axis=2)[:, :, : ,None], [1, 1, 1, 2]))),
        ('...ij, ...jc -> ...ic', [(2, 1, 5, 4), (2, 1, 4, 2)], lambda *args: (
                                                            onp.tile(args[1].sum(axis=3)[:, :, None, :], [1, 1, 5, 1]),
                                                             onp.tile(args[0].sum(axis=2)[:, :, : ,None], [1, 1, 1, 2]))),
        # test with cuTensor using workspace
        (('ij,jk,kl->il'), [(64, 200), (200, 64), (64, 64)],
            lambda *args: (onp.dot(onp.ones((64, 64)), onp.dot(args[1], args[2]).T),
            onp.dot(args[0].T, onp.dot(onp.ones((64, 64)), args[2].T)),
            onp.dot(onp.dot(args[0], args[1]).T, onp.ones((64, 64)))))
    ]

    dtypes = ['float16', 'float32', 'float64', 'int32']
    for hybridize in [False, True]:
        for cache_setting in ['0', '1', None]:
            for dtype in dtypes:
                for config in configs:
                    for optimize in [False, True]:
                        with environment('MXNET_CUTENSOR_CACHEFILE', cache_setting):
                            rtol = 1e-1 if dtype == 'float16' else 1e-3
                            atol = 1e-1 if dtype == 'float16' else 1e-4
                            (subscripts, operands, get_grad) = config
                            test_einsum = TestEinsum(subscripts, optimize)
                            if hybridize:
                                test_einsum.hybridize()
                            x = []
                            x_np = []
                            for shape in operands:
                                tmp = onp.array(onp.random.uniform(-0.3, 0.3, shape), dtype=dtype)
                                x_np.append(tmp)
                                x.append(np.array(tmp, dtype=dtype))
                                x[-1].attach_grad()
                            expected_np = onp.einsum(subscripts, *x_np, optimize=False, dtype=dtype).astype(dtype)
                            with mx.autograd.record():
                                out_mx = test_einsum(*x)
                            assert out_mx.shape == expected_np.shape
                            assert_almost_equal(out_mx.asnumpy(), expected_np, rtol=rtol, atol=atol)
                            out_mx.backward()
                            for (iop, op) in enumerate(x):
                                assert_almost_equal(op.grad.asnumpy(), get_grad(*x_np)[iop], rtol=rtol, atol=atol)
