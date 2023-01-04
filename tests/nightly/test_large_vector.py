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

import os
import sys
import math
import numpy as np
import mxnet as mx

from mxnet.test_utils import rand_ndarray, assert_almost_equal, rand_coord_2d, create_vector
from mxnet.util import TemporaryDirectory
from mxnet import gluon, nd
from common import with_seed
import pytest


# dimension constants
LARGE_X = 4300000000
MEDIUM_X = 1000000000


@pytest.mark.timeout(0)
def test_nn():
    def check_dense():
        data = mx.nd.ones(shape=LARGE_X)
        linear = gluon.nn.Dense(2)
        linear.initialize()
        res = linear(data)
        assert res.shape == (LARGE_X, 2)

    def check_sign():
        a = mx.nd.random.normal(-1, 1, shape=LARGE_X)
        mx_res = mx.nd.sign(a)
        assert_almost_equal(mx_res[-1].asnumpy(), np.sign(a[-1].asnumpy()))

    # TODO: correctness of layernorm
    # numpy implementation for large vector is flaky
    def check_layer_norm():
        axis = 0
        eps = 1E-5
        in_shape = LARGE_X
        data = nd.random.normal(0, 1, in_shape)
        gamma = nd.random.normal(0, 1, in_shape)
        beta = nd.random.normal(0, 1, in_shape)
        mx_out = nd.LayerNorm(data, gamma, beta, axis, eps)
        assert mx_out.shape == (in_shape,)

    # TODO: correctness of batchnorm
    # in future, we could test if mean, var of output
    # matches target output's mean, var
    def check_batchnorm():
        shape = LARGE_X
        axis = 0  # since vector
        data = mx.nd.ones(shape=shape)
        bn_gamma = mx.nd.random.uniform(shape=shape)
        bn_beta = mx.nd.random.uniform(shape=shape)
        bn_running_mean = mx.nd.zeros(shape)
        bn_running_var = mx.nd.ones(shape)
        output = mx.nd.BatchNorm(data, bn_gamma, bn_beta,
                                 bn_running_mean, bn_running_var, axis=axis)
        assert output.shape == (shape,)

    def check_sequence_mask():
        # Sequence Mask input [max_sequence_length, batch_size]
        # test with input batch_size = 2
        a = nd.arange(0, LARGE_X * 2).reshape(LARGE_X, 2)
        # test as identity operator
        b = nd.SequenceMask(a)
        assert b[-1][0] == a[-1][0]
        assert b.shape == a.shape
        # test with default mask
        b = nd.SequenceMask(a, sequence_length=nd.array([1, 1]),
                            use_sequence_length=True)
        assert b[0][1] == a[0][1]  # first sequence of each batch kept
        assert b[-1][-1] != a[-1][-1]  # rest sequences masked
        assert b[-1][-1] == 0
        # test with mask value
        b = nd.SequenceMask(a, sequence_length=nd.array([1, 1]),
                            use_sequence_length=True, value=-1)
        assert b[-1][-1] == -1

    def check_sequence_reverse():
        a = nd.arange(0, LARGE_X * 2).reshape(LARGE_X, 2)
        # test as reverse operator
        b = nd.SequenceReverse(a)
        assert b[-1][0] == a[0][0]
        assert b.shape == a.shape
        # test with sequence length
        b = nd.SequenceReverse(a, sequence_length=nd.array([2, 3]),
                               use_sequence_length=True)
        assert b[1][0] == a[0][0]  # check if reversed
        assert b[-1][0] == a[-1][0]  # check if intact
        assert b.shape == a.shape

    def check_sequence_last():
        a = nd.arange(0, LARGE_X * 2).reshape(LARGE_X, 2)
        # test if returns last sequence
        b = nd.SequenceLast(a)
        assert_almost_equal(b.asnumpy(), a[-1].asnumpy())
        assert b.shape == (2,)
        # test with sequence length
        # parameter sequence_length - NDArray with shape (batch_size)
        # (2,3) indicates 2nd sequence from batch 1 and 3rd sequence from batch 2
        # need to mention dtype = int64 for sequence_length ndarray to support large indices
        # else it defaults to float32 and errors
        b = nd.SequenceLast(a, sequence_length=mx.nd.array([2, 3], dtype="int64"),
                            use_sequence_length=True)
        # check if it takes 2nd sequence from the first batch
        assert b[0] == a[1][0]

    check_sequence_last()
    check_dense()
    check_sign()
    check_layer_norm()
    check_batchnorm()
    check_sequence_mask()
    check_sequence_reverse()


@pytest.mark.timeout(0)
def test_tensor():
    def check_ndarray_zeros():
        a = nd.zeros(shape=LARGE_X)
        assert a[-1] == 0
        assert a.shape == (LARGE_X,)
        assert a.size == LARGE_X

    def check_ndarray_ones():
        a = nd.ones(shape=LARGE_X)
        assert a[-1] == 1
        assert nd.sum(a) == LARGE_X

    def check_ndarray_empty():
        a = nd.empty(LARGE_X)
        assert a.shape == (LARGE_X,)

    @with_seed()
    def check_ndarray_random_uniform():
        a = nd.random.uniform(shape=LARGE_X)
        assert a[-1] != 0

    @pytest.mark.skip(reason="Randint flaky, tracked at https://github.com/apache/mxnet/issues/16172")
    @with_seed()
    def check_ndarray_random_randint():
        # check if randint can generate value greater than 2**32 (large)
        low = 2**32
        high = 2**34
        a = nd.random.randint(low, high, dtype=np.int64, shape=LARGE_X).asnumpy()
        assert a.shape == (LARGE_X,)
        assert (a >= low).all() and (a < high).all()

    @with_seed()
    def check_ndarray_random_exponential():
        a = nd.random.exponential(shape=LARGE_X)
        assert a[-1] >= 0.
        assert a.shape[0] == LARGE_X

    @with_seed()
    def check_ndarray_random_gamma():
        a = nd.random.gamma(shape=LARGE_X)
        assert a[-1] >= 0.
        assert a.shape[0] == LARGE_X

    @with_seed()
    def check_ndarray_random_generalized_negative_binomial():
        a = nd.random.generalized_negative_binomial(shape=LARGE_X)
        assert a[-1] >= 0.
        assert a.shape[0] == LARGE_X

    @with_seed()
    def check_ndarray_random_multinomial():
        a = nd.random.multinomial(nd.random.uniform(shape=LARGE_X))
        assert a[-1] >= 0.
        assert a.shape[0] == 1

    @with_seed()
    def check_ndarray_random_negative_binomial():
        a = nd.random.negative_binomial(shape=LARGE_X)
        assert a[-1] >= 0.
        assert a.shape[0] == LARGE_X

    @with_seed()
    def check_ndarray_random_normal():
        a = nd.random.normal(shape=LARGE_X)
        assert a.shape[0] == LARGE_X

    @with_seed()
    def check_ndarray_random_poisson():
        a = nd.random.poisson(shape=LARGE_X)
        assert a[-1] >= 0.
        assert a.shape[0] == LARGE_X

    @with_seed()
    def check_ndarray_random_randn():
        a = nd.random.randn(LARGE_X)
        assert a.shape[0] == LARGE_X

    @with_seed()
    def check_ndarray_random_shuffle():
        a = nd.ones(shape=LARGE_X)
        a[-1] = 3
        a = nd.random.shuffle(a)
        unique_a = np.unique(a.asnumpy())
        assert len(unique_a) == 2  # only 2 unique values
        assert unique_a[0] == 1  # first unique value is 1
        assert unique_a[1] == 3  # second unique value is 3
        assert a.shape[0] == LARGE_X

    def check_full():
        a = nd.full(LARGE_X, 3)
        assert a.shape[0] == LARGE_X
        assert a[LARGE_X // 2] == 3
        assert a[-1] == 3

    def check_repeat():
        x = create_vector(size=LARGE_X//2)
        y = nd.repeat(x, repeats=2, axis = 0)
        assert y.shape[0] == LARGE_X
        assert y[1] == 0
        assert y[LARGE_X-1] == LARGE_X//2-1

    def check_clip():
        a = create_vector(LARGE_X)
        res = nd.clip(a, a_min=100, a_max=1000)
        assert res[-1] == 1000

    def check_slice():
        a = nd.ones(LARGE_X)
        res = nd.slice(a, begin=(LARGE_X - MEDIUM_X), end=LARGE_X)
        assert res.shape[0] == MEDIUM_X
        assert res[0] == 1

    def check_slice_assign():
        a = nd.ones(shape=LARGE_X)
        a[LARGE_X-1:LARGE_X] = 1000
        assert np.sum(a[-1].asnumpy() == 1000) == 1

    def check_take():
        a = nd.ones(shape=LARGE_X)
        idx = nd.arange(LARGE_X - 1000, LARGE_X)
        res = nd.take(a, idx)
        assert np.sum(res.asnumpy() == 1) == res.shape[0]

    def check_expand_dims():
        a = nd.ones(shape=LARGE_X)
        res = nd.expand_dims(a, axis=0)
        assert res[0][0] == 1
        assert res.shape == (1, a.shape[0])

    def check_squeeze():
        a = nd.ones(shape=LARGE_X)
        data = nd.expand_dims(a, axis=0)
        res = nd.squeeze(data)
        assert a[0] == res[0]
        assert res.shape == a.shape

    def check_broadcast_div():
        a = nd.ones(shape=LARGE_X)
        b = nd.ones(shape=LARGE_X) * 2
        res = a / b
        assert np.sum(res.asnumpy() == 0.5) == a.shape[0]

    def check_size():
        b = create_vector(size=LARGE_X)
        # explicit wait_to_read()
        assert b[0] == 0
        assert b.size == LARGE_X

    def check_copy():
        a = nd.ones(LARGE_X)
        b = a.copy()
        assert a[0] == b[0]
        assert b.shape == a.shape
        assert b.size == LARGE_X

    def check_copy_to():
        a = create_vector(size=LARGE_X)
        # keeping dtype same as input uses parallel copy which is much faster
        b = nd.zeros(LARGE_X, dtype=np.int64)
        c = a.copyto(b)
        assert c is b
        assert b[-1] == LARGE_X-1
        assert b[0] == 0

    def check_zeros_like():
        a = nd.ones(LARGE_X)
        b = nd.zeros_like(a)
        assert b[-1] == 0
        assert b.shape == a.shape

    def check_ones_like():
        a = nd.zeros(LARGE_X)
        b = nd.ones_like(a)
        assert b[-1] == 1
        assert b.shape == a.shape

    def check_shape():
        b = create_vector(size=LARGE_X)
        # explicit wait_to_read()
        assert b[0] == 0
        assert b.shape[0] == LARGE_X

    def check_concat():
        a = nd.ones(LARGE_X)
        b = nd.zeros(LARGE_X)
        c = nd.concat(a, b, dim=0)
        assert c[0] == 1
        assert c[-1] == 0
        assert c.shape[0] == (2 * LARGE_X)

    def check_slice_like():
        a = create_vector(size=LARGE_X)
        b = nd.ones(LARGE_X//2)
        c = nd.slice_like(a, b)
        assert c.shape == b.shape
        assert c[0] == 0
        assert c[-1] == (LARGE_X // 2 - 1)

    def check_slice_axis():
        a = create_vector(size=LARGE_X)
        med = LARGE_X // 2
        c = nd.slice_axis(a, axis=0, begin=0, end=med)
        assert c.shape[0] == a.shape[0] // 2
        assert c[-1][0] == (med - 1)

    def check_gather():
        arr = mx.nd.ones(LARGE_X)
        # Passing dtype=np.int64 since randomly generated indices are
        # very large that exceeds int32 limits.
        idx = mx.nd.random.randint(0, LARGE_X, 10, dtype=np.int64)
        # Calls gather_nd internally
        tmp = arr[idx]
        assert np.sum(tmp.asnumpy() == 1) == 10
        # Calls gather_nd internally
        arr[idx] += 1
        assert np.sum(arr[idx].asnumpy() == 2) == 10

    def check_infer_shape():
        data_1 = mx.symbol.Variable('data_1')
        data_2 = mx.symbol.Variable('data_2')
        add = data_1+data_2
        # > add.infer_shape(data_1=(LARGE_X,), data_2=(LARGE_X,))
        # OUTPUT - arg_shapes, out_shapes, aux_shapes
        _, out_shapes, _ = add.infer_shape(data_1=(LARGE_X,), data_2=(LARGE_X,))
        assert out_shapes == [(LARGE_X,)]

    def check_astype():
        x = create_vector(size=LARGE_X//4)
        x = nd.tile(x, 4)
        y = x.astype('int32')
        assert y.dtype == np.int32
        assert y[-1] == LARGE_X//4-1

    def check_cast():
        x = create_vector(size=LARGE_X//4)
        x = nd.tile(x, 4)
        y = nd.cast(x, np.int32)
        assert y.dtype == np.int32
        assert y[-1] == LARGE_X//4-1

    def check_load_save():
        x = create_vector(size=LARGE_X)
        with TemporaryDirectory() as tmp:
            tmpfile = os.path.join(tmp, 'large_vector')
            nd.save(tmpfile, [x])
            y = nd.load(tmpfile)
            y = y[0]
            assert x[0] == y[0]
            assert x[-1] == y[-1]

    def check_binary_broadcast():
        def check_correctness(mxnet_op, numpy_op, atol=1e-3):
            a = mx.nd.ones(LARGE_X).as_np_ndarray()
            b = 2*mx.nd.ones(LARGE_X).as_np_ndarray()
            res = mxnet_op(a, b)
            np_res = numpy_op(1, 2)
            assert np.abs(res[-1] - np_res) < atol
        check_correctness(mx.np.arctan2, np.arctan2)
        check_correctness(mx.np.hypot, np.hypot)

    check_ndarray_zeros()
    check_ndarray_ones()
    check_ndarray_empty()
    check_ndarray_random_uniform()
    check_ndarray_random_randint()
    check_ndarray_random_exponential()
    check_ndarray_random_gamma()
    check_ndarray_random_generalized_negative_binomial()
    check_ndarray_random_multinomial()
    check_ndarray_random_negative_binomial()
    check_ndarray_random_normal()
    check_ndarray_random_poisson()
    check_ndarray_random_randn()
    check_ndarray_random_shuffle()
    check_full()
    check_repeat()
    check_clip()
    check_slice()
    check_slice_assign()
    check_take()
    check_expand_dims()
    check_squeeze()
    check_broadcast_div()
    check_size()
    check_copy()
    check_copy_to()
    check_zeros_like()
    check_ones_like()
    check_shape()
    check_concat()
    check_slice_like()
    check_slice_axis()
    check_gather()
    check_infer_shape()
    check_astype()
    check_cast()
    check_load_save()
    check_binary_broadcast()


@pytest.mark.timeout(0)
def test_basic():
    def check_elementwise():
        a = nd.ones(shape=LARGE_X)
        b = nd.ones(shape=LARGE_X)
        res = a + b
        assert res[-1].asnumpy() == 2
        res = a + 1
        assert res[-1].asnumpy() == 2
        res = nd.sqrt(a + 8)
        assert res[-1].asnumpy() == 3

    def check_argmin():
        a = create_vector(LARGE_X, dtype=np.float32)
        assert a[0] == 0
        idx = mx.nd.argmin(a, axis=0)
        assert idx[0] == 0
        assert idx.shape[0] == 1

    @pytest.mark.skip(reason="Memory doesn't free up after stacked execution with other ops, " +
                      "tracked at https://github.com/apache/mxnet/issues/17411")
    def check_argsort():
        a = create_vector(size=LARGE_X)
        s = nd.argsort(a, axis=0, is_ascend=False, dtype=np.int64)
        assert s[0] == (LARGE_X - 1)

    @pytest.mark.skip(reason="Memory doesn't free up after stacked execution with other ops, " +
                      "tracked at https://github.com/apache/mxnet/issues/17411")
    def check_sort():
        a = create_vector(size=LARGE_X)

        def check_descend(x):
            s = nd.sort(x, axis=0, is_ascend=False)
            assert s[-1] == 0

        def check_ascend(x):
            s = nd.sort(x, is_ascend=True)
            assert s[0] == 0

        check_descend(a)
        check_ascend(a)

    @pytest.mark.skip(reason="Memory doesn't free up after stacked execution with other ops, " +
                      "tracked at https://github.com/apache/mxnet/issues/17411")
    def check_topk():
        a = create_vector(size=LARGE_X)
        ind = nd.topk(a, k=10, axis=0, dtype=np.int64)
        for i in range(10):
            assert ind[i] == (LARGE_X - i - 1)
        ind, val = mx.nd.topk(a, k=3, axis=0, dtype=np.int64, ret_typ="both", is_ascend=False)
        assert np.all(ind == val)
        val = nd.topk(a, k=1, axis=0, dtype=np.int64, ret_typ="value")
        assert val == (LARGE_X - 1)

    def check_mean():
        a = nd.arange(-LARGE_X // 2, LARGE_X // 2 + 1, dtype=np.int64)
        b = nd.mean(a, axis=0)
        assert b == 0

    def check_exponent_logarithm_operators():
        a = 2*nd.ones(shape=LARGE_X)
        # exponent
        result = nd.exp(a)
        assert result[-1] == 7.389056
        assert result.shape == a.shape
        # exponent minus 1
        result = nd.expm1(a)
        assert result[-1] == 6.389056
        assert result.shape == a.shape
        # log2
        result = nd.log2(a)
        assert result[-1] == 1
        assert result.shape == a.shape
        # log10
        result = nd.log10(a)
        assert result[-1] == 0.30103
        assert result.shape == a.shape
        # log1p
        result = nd.log1p(a)
        assert result[-1] == 1.0986123
        assert result.shape == a.shape
        # log
        result = nd.log(a)
        assert result[-1] == 0.6931472
        assert result.shape == a.shape

    def check_power_operators():
        a = 2*nd.ones(shape=LARGE_X)
        # sqrt
        result = nd.sqrt(a)
        assert result[-1] == 1.4142135
        assert result.shape == a.shape
        # rsqrt
        result = nd.rsqrt(a)
        assert result[-1] == 0.70710677
        assert result.shape == a.shape
        # cbrt
        result = nd.cbrt(a)
        assert result[-1] == 1.2599211
        assert result.shape == a.shape
        # rcbrt
        result = nd.rcbrt(a)
        assert result[-1] == 0.7937005
        assert result.shape == a.shape
        # square
        result = nd.square(a)
        assert result[-1] == 4
        assert result.shape == a.shape
        # reciprocal
        result = nd.reciprocal(a)
        assert result[-1] == 0.5
        assert result.shape == a.shape

    def check_add():
        a = nd.ones(shape=LARGE_X)
        b = nd.ones(shape=LARGE_X)
        c = b
        c = c.__add__(a)
        assert c[-1] == 2
        assert c.shape == a.shape

    def check_sub():
        a = 3*nd.ones(shape=LARGE_X)
        b = nd.ones(shape=LARGE_X)
        c = b
        c = c.__sub__(a)
        assert c[-1] == -2
        assert c.shape == a.shape

    def check_rsub():
        a = 3*nd.ones(shape=LARGE_X)
        b = nd.ones(shape=LARGE_X)
        c = b
        c = c.__rsub__(a)
        assert c[-1] == 2
        assert c.shape == a.shape

    def check_neg():
        a = nd.ones(shape=LARGE_X)
        c = a
        c = c.__neg__()
        assert c[-1] == -1
        assert c.shape == a.shape

    def check_mul():
        a = 2*nd.ones(shape=LARGE_X)
        b = 3*nd.ones(shape=LARGE_X)
        c = b
        c = c.__mul__(a)
        assert c[-1] == 6
        assert c.shape == a.shape

    def check_div():
        a = 2*nd.ones(shape=LARGE_X)
        b = 3*nd.ones(shape=LARGE_X)
        c = b
        c = c.__div__(a)
        assert c[-1] == 3/2
        assert c.shape == a.shape

    def check_rdiv():
        a = 2*nd.ones(shape=LARGE_X)
        b = 3*nd.ones(shape=LARGE_X)
        c = b
        c = c.__rdiv__(a)
        assert c[-1] == 2/3
        assert c.shape == a.shape

    def check_mod():
        a = 2*nd.ones(shape=LARGE_X)
        b = 3*nd.ones(shape=LARGE_X)
        c = b
        c = c.__mod__(a)
        assert c[-1] == 1
        assert c.shape == a.shape

    def check_rmod():
        a = 2*nd.ones(shape=LARGE_X)
        b = 3*nd.ones(shape=LARGE_X)
        c = b
        c = c.__rmod__(a)
        assert c[-1] == 2
        assert c.shape == a.shape

    def check_imod():
        a = 2*nd.ones(shape=LARGE_X)
        b = 3*nd.ones(shape=LARGE_X)
        c = b
        c = c.__imod__(a)
        assert c[-1] == 1
        assert c.shape == a.shape

    def check_pow():
        a = 2*nd.ones(shape=LARGE_X)
        b = 3*nd.ones(shape=LARGE_X)
        c = b
        c = c.__pow__(a)
        assert c[-1] == 9
        assert c.shape == a.shape

    def check_rpow():
        a = 2*nd.ones(shape=LARGE_X)
        b = 3*nd.ones(shape=LARGE_X)
        c = b
        c = c.__rpow__(a)
        assert c[-1] == 8
        assert c.shape == a.shape

    def check_sum():
        a = nd.ones(LARGE_X)
        b = nd.sum(a, axis=0)
        assert b[0] == LARGE_X

    def check_prod():
        a = nd.ones(LARGE_X)
        b = nd.prod(a, axis=0)
        assert b[0] == 1

    def check_min():
        a = create_vector(size=LARGE_X)
        b = nd.min(a, axis=0)
        assert b[0] == 0
        assert b[-1] == 0

    def check_max():
        a = create_vector(size=LARGE_X)
        b = nd.max(a, axis=0)
        assert b[0] == (LARGE_X - 1)

    def check_argmax():
        a = nd.ones(LARGE_X)
        b = nd.zeros(LARGE_X)
        c = nd.concat(a, b, dim=0)
        d = nd.argmax(c, axis=0)
        assert c.shape[0] == (2 * LARGE_X)
        assert d == 0

    def check_iadd():
        a = nd.ones(LARGE_X)
        b = nd.ones(LARGE_X)
        c = b
        c += a
        assert c.shape == a.shape
        assert c[-1] == 2

    def check_isub():
        a = nd.full(LARGE_X, 3)
        b = nd.ones(LARGE_X)
        c = a
        c -= b
        assert c.shape == a.shape
        assert c[-1] == 2

    def check_imul():
        a = nd.full(LARGE_X, 3)
        b = nd.ones(LARGE_X)
        c = b
        c *= a
        assert c.shape == a.shape
        assert c[-1] == 3

    def check_idiv():
        a = nd.full(LARGE_X, 4)
        b = nd.full(LARGE_X, 2)
        c = a
        c /= b
        assert c.shape == a.shape
        assert c[-1] == 2

    def check_eq():
        a = nd.full(LARGE_X, 3)
        b = nd.full(LARGE_X, 3)
        c = (a == b)
        assert (c.asnumpy() == 1).all()

    def check_neq():
        a = nd.full(LARGE_X, 2)
        b = nd.full(LARGE_X, 3)
        c = (a != b)
        assert (c.asnumpy() == 1).all()

    def check_lt():
        a = nd.full(LARGE_X, 2)
        b = nd.full(LARGE_X, 3)
        d = (a <= b)
        assert (d.asnumpy() == 1).all()

    def check_lte():
        a = nd.full(LARGE_X, 2)
        b = nd.full(LARGE_X, 3)
        c = nd.full(LARGE_X, 2)
        d = (a <= b)
        assert (d.asnumpy() == 1).all()
        d = (a <= c)
        assert (d.asnumpy() == 1).all()

    def check_gt():
        a = nd.full(LARGE_X, 3)
        b = nd.full(LARGE_X, 2)
        d = (a > b)
        assert (d.asnumpy() == 1).all()

    def check_gte():
        a = nd.full(LARGE_X, 3)
        b = nd.full(LARGE_X, 2)
        c = nd.full(LARGE_X, 3)
        d = (a >= b)
        assert (d.asnumpy() == 1).all()
        d = (a >= c)
        assert (d.asnumpy() == 1).all()

    def check_logical():
        def check_logical_and(a, b):
            mx_res = mx.nd.logical_and(a, b)
            assert_almost_equal(mx_res[-1].asnumpy(), np.logical_and(a[-1].asnumpy(), b[-1].asnumpy()))

        def check_logical_or(a, b):
            mx_res = mx.nd.logical_or(a, b)
            assert_almost_equal(mx_res[-1].asnumpy(), np.logical_or(a[-1].asnumpy(), b[-1].asnumpy()))

        def check_logical_not(a, b):
            mx_res = mx.nd.logical_not(a, b)
            assert_almost_equal(mx_res[-1].asnumpy(), np.logical_not(a[-1].asnumpy(), b[-1].asnumpy()))

        def check_logical_xor(a, b):
            mx_res = mx.nd.logical_xor(a, b)
            assert_almost_equal(mx_res[-1].asnumpy(), np.logical_xor(a[-1].asnumpy(), b[-1].asnumpy()))

        a = mx.nd.ones(LARGE_X)
        b = mx.nd.zeros(LARGE_X)
        check_logical_and(a, b)
        check_logical_or(a, b)
        check_logical_not(a, b)
        check_logical_xor(a, b)

    def create_input_for_rounding_ops():
        # Creates an vector with values (-LARGE/2 .... -2, -1, 0, 1, 2, .... , LARGE/2-1)
        # then divides each element by 2 i.e (-LARGE/4 .... -1, -0.5, 0, 0.5, 1, .... , LARGE/4-1)
        inp = nd.arange(-LARGE_X//2, LARGE_X//2, dtype=np.float64)
        inp = inp/2
        return inp

    def assert_correctness_of_rounding_ops(output, mid, expected_vals):
        # checks verifies 5 values at the middle positions of the input vector
        # i.e mid-2, mid-1, mid, mid+1, mid+2
        output_idx_to_inspect = [mid-2, mid-1, mid, mid+1, mid+2]
        for i in range(len(output_idx_to_inspect)):
            assert output[output_idx_to_inspect[i]] == expected_vals[i]

    def check_rounding_ops():
        x = create_input_for_rounding_ops()

        def check_ceil():
            y = nd.ceil(x)
            # expected ouput for middle 5 values after applying ceil()
            expected_output = [-1, 0, 0, 1, 1]
            assert_correctness_of_rounding_ops(y, LARGE_X//2, expected_output)

        def check_fix():
            y = nd.fix(x)
            # expected ouput for middle 5 values after applying fix()
            expected_output = [-1, 0, 0, 0, 1]
            assert_correctness_of_rounding_ops(y, LARGE_X//2, expected_output)

        def check_floor():
            y = nd.floor(x)
            # expected ouput for middle 5 values after applying floor()
            expected_output = [-1, -1, 0, 0, 1]
            assert_correctness_of_rounding_ops(y, LARGE_X//2, expected_output)

        def check_rint():
            y = nd.rint(x)
            # expected ouput for middle 5 values after applying rint()
            expected_output = [-1, -1, 0, 0, 1]
            assert_correctness_of_rounding_ops(y, LARGE_X//2, expected_output)

        def check_round():
            y = nd.round(x)
            # expected ouput for middle 5 values after applying round()
            expected_output = [-1, -1, 0, 1, 1]
            assert_correctness_of_rounding_ops(y, LARGE_X//2, expected_output)

        def check_trunc():
            y = nd.trunc(x)
            # expected ouput for middle 5 values after applying trunc()
            expected_output = [-1, 0, 0, 0, 1]
            assert_correctness_of_rounding_ops(y, LARGE_X//2, expected_output)

        check_ceil()
        check_fix()
        check_floor()
        check_rint()
        check_round()
        check_trunc()

    def create_input_for_trigonometric_ops(vals):
        # Creates large vector input of size(LARGE_X) from vals using tile operator
        inp = nd.array(vals)
        inp = nd.tile(inp, LARGE_X//len(vals))
        return inp

    def assert_correctness_of_trigonometric_ops(output, expected_vals):
        # checks verifies 5 values at positions(0, 1, -3, -2, -1) of the input vector
        output_idx_to_inspect = [0, 1, -3, -2, -1]
        for i in range(len(output_idx_to_inspect)):
            assert np.abs(output[output_idx_to_inspect[i]].asnumpy()-expected_vals[i]) <= 1e-3

    def check_trigonometric_ops():
        def check_arcsin():
            x = create_input_for_trigonometric_ops([-1, -.707, 0, .707, 1])
            y = nd.arcsin(x)
            # expected ouput for indices=(0, 1, -3, -2, -1) after applying arcsin()
            expected_output = [-np.pi/2, -np.pi/4, 0, np.pi/4, np.pi/2]
            assert_correctness_of_trigonometric_ops(y, expected_output)

        def check_arccos():
            x = create_input_for_trigonometric_ops([-1, -.707, 0, .707, 1])
            y = nd.arccos(x)
            # expected ouput for indices=(0, 1, -3, -2, -1) after applying arccos()
            expected_output = [np.pi, 3*np.pi/4, np.pi/2, np.pi/4, 0]
            assert_correctness_of_trigonometric_ops(y, expected_output)

        def check_arctan():
            x = create_input_for_trigonometric_ops([-np.Inf, -1, 0, 1, np.Inf])
            y = nd.arctan(x)
            # expected ouput for indices=(0, 1, -3, -2, -1) after applying arctan()
            expected_output = [-np.pi/2, -np.pi/4, 0, np.pi/4, np.pi/2]
            assert_correctness_of_trigonometric_ops(y, expected_output)

        def check_sin():
            x = create_input_for_trigonometric_ops([-np.pi/2, -np.pi/4, 0, np.pi/4, np.pi/2])
            y = nd.sin(x)
            # expected ouput for indices=(0, 1, -3, -2, -1) after applying sin()
            expected_output = [-1, -.707, 0, .707, 1]
            assert_correctness_of_trigonometric_ops(y, expected_output)

        def check_cos():
            x = create_input_for_trigonometric_ops([0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi])
            y = nd.cos(x)
            # expected ouput for indices=(0, 1, -3, -2, -1) after applying cos()
            expected_output = [1, .707, 0, -.707, -1]
            assert_correctness_of_trigonometric_ops(y, expected_output)

        def check_tan():
            x = create_input_for_trigonometric_ops([-np.pi/6, -np.pi/4, 0, np.pi/4, np.pi/6])
            y = nd.tan(x)
            # expected ouput for indices=(0, 1, -3, -2, -1) after applying tan()
            expected_output = [-.577, -1, 0, 1, .577]
            assert_correctness_of_trigonometric_ops(y, expected_output)

        def check_arcsinh():
            x = create_input_for_trigonometric_ops([-np.pi/2, -np.pi/4, 0, np.pi/4, np.pi/2])
            y = nd.arcsinh(x)
            # expected ouput for indices=(0, 1, -3, -2, -1) after applying arcsinh()
            expected_output = [np.arcsinh(-np.pi/2), np.arcsinh(-np.pi/4), 0, np.arcsinh(np.pi/4), np.arcsinh(np.pi/2)]
            assert_correctness_of_trigonometric_ops(y, expected_output)

        def check_arccosh():
            x = create_input_for_trigonometric_ops([1, np.pi/2, 3*np.pi/4, np.pi, 5*np.pi/4])
            y = nd.arccosh(x)
            # expected ouput for indices=(0, 1, -3, -2, -1) after applying arccosh()
            expected_output = [0, np.arccosh(np.pi/2), np.arccosh(3*np.pi/4), np.arccosh(np.pi), np.arccosh(5*np.pi/4)]
            assert_correctness_of_trigonometric_ops(y, expected_output)

        def check_arctanh():
            x = create_input_for_trigonometric_ops([-1/4, -1/2, 0, 1/4, 1/2])
            y = nd.arctanh(x)
            # expected ouput for indices=(0, 1, -3, -2, -1) after applying arctanh()
            expected_output = [np.arctanh(-1/4), np.arctanh(-1/2), 0, np.arctanh(1/4), np.arctanh(1/2)]
            assert_correctness_of_trigonometric_ops(y, expected_output)

        def check_sinh():
            x = create_input_for_trigonometric_ops([-np.pi/2, -np.pi/4, 0, np.pi/4, np.pi/2])
            y = nd.sinh(x)
            # expected ouput for indices=(0, 1, -3, -2, -1) after applying sinh()
            expected_output = [np.sinh(-np.pi/2), np.sinh(-np.pi/4), 0, np.sinh(np.pi/4), np.sinh(np.pi/2)]
            assert_correctness_of_trigonometric_ops(y, expected_output)

        def check_cosh():
            x = create_input_for_trigonometric_ops([0, 1, np.pi/2, 3*np.pi/4, np.pi])
            y = nd.cosh(x)
            # expected ouput for indices=(0, 1, -3, -2, -1) after applying cosh()
            expected_output = [1, np.cosh(1), np.cosh(np.pi/2), np.cosh(3*np.pi/4), np.cosh(np.pi)]
            assert_correctness_of_trigonometric_ops(y, expected_output)

        def check_tanh():
            x = create_input_for_trigonometric_ops([-1/4, -1/2, 0, 1/4, 1/2])
            y = nd.tanh(x)
            # expected ouput for indices=(0, 1, -3, -2, -1) after applying tanh()
            expected_output = [np.tanh(-1/4), np.tanh(-1/2), 0, np.tanh(1/4), np.tanh(1/2)]
            assert_correctness_of_trigonometric_ops(y, expected_output)

        def check_radians():
            x = create_input_for_trigonometric_ops([0, 90, 180, 270, 360])
            y = nd.radians(x)
            # expected ouput for indices=(0, 1, -3, -2, -1) after applying radians()
            expected_output = [0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi]
            assert_correctness_of_trigonometric_ops(y, expected_output)

        def check_degrees():
            x = create_input_for_trigonometric_ops([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
            y = nd.degrees(x)
            # expected ouput for indices=(0, 1, -3, -2, -1) after applying degrees()
            expected_output = [0, 90, 180, 270, 360]
            assert_correctness_of_trigonometric_ops(y, expected_output)

        check_arcsin()
        check_arccos()
        check_arctan()
        check_sin()
        check_cos()
        check_tan()
        check_arcsinh()
        check_arccosh()
        check_arctanh()
        check_sinh()
        check_cosh()
        check_tanh()
        check_radians()
        check_degrees()

    def check_add_n():
        x = [nd.ones(LARGE_X)]
        y = nd.add_n(*x)
        assert y[0] == 1
        assert y[-1] == 1

    def check_modulo():
        x = mx.nd.ones(LARGE_X)*6
        y = mx.nd.ones(LARGE_X)*4
        z = (x % y)
        assert z[0] == 2
        assert z[-1] == 2
        x = mx.nd.ones(LARGE_X)*5
        z = nd.modulo(x, y)
        assert z[0] == 1
        assert z[-1] == 1

    def check_maximum():
        x = mx.nd.ones(LARGE_X)*3
        y = mx.nd.ones(LARGE_X)*4
        z = nd.maximum(x, y)
        assert z[0] == 4
        assert z[-1] == 4
        z = nd.maximum(x, 5)
        assert z[0] == 5
        assert z[-1] == 5

    def check_minimum():
        x = mx.nd.ones(LARGE_X)*3
        y = mx.nd.ones(LARGE_X)*2
        z = nd.minimum(x, y)
        assert z[0] == 2
        assert z[-1] == 2
        z = nd.minimum(x, 5)
        assert z[0] == 3
        assert z[-1] == 3

    check_elementwise()
    check_argmin()
    check_argsort()
    check_sort()
    check_topk()
    check_mean()
    check_exponent_logarithm_operators()
    check_power_operators()
    check_add()
    check_sub()
    check_rsub()
    check_neg()
    check_mul()
    check_div()
    check_rdiv()
    check_mod()
    check_rmod()
    check_imod()
    check_pow()
    check_rpow()
    check_sum()
    check_prod()
    check_min()
    check_max()
    check_argmax()
    check_iadd()
    check_isub()
    check_imul()
    check_idiv()
    check_eq()
    check_neq()
    check_lt()
    check_lte()
    check_gt()
    check_gte()
    check_logical()
    check_rounding_ops()
    check_trigonometric_ops()
    check_add_n()
    check_modulo()
    check_maximum()
    check_minimum()

