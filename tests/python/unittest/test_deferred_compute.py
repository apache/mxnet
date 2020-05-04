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

import functools
import operator

import numpy as np

import mxnet as mx
import mxnet._deferred_compute as dc
from mxnet.base import MXNetError
import pytest


def _all_same(arrays1, arrays2, message=''):
    same = all(np.array_equal(a1, a2) for a1, a2 in zip(arrays1, arrays2))
    if not same:
        raise AssertionError('Arrays not equal ({}):\n{}\n\n{}'.format(message, arrays1, arrays2))


def _assert_dc(setup, compute, mode='all', setup_is_deterministic=True, numpy=True):
    """Compare results of deferred compute and normal imperative mode.

    Parameters
    ----------
    setup : callable
        Setup function computing inputs for compute function. Always called
        outside of deferred compute.
    compute : callable
        Compute function. We compare the output between normal computation and
        deferred compute.
    mode : {'all', 'symbolic', 'imperative', 'imperativewithnondccompute'}
        Compare deferred compute outputs triggered via imperative computation
        (eg. asnumpy() conversion) or obtained from the exported symbol or
        both.
    setup_is_deterministic : bool
        If True, setup function may be called multiple times. If False, will
        only be called once.
    numpy : bool
        If True, use mx.np. Otherwise mx.nd.

    """
    try:
        nd = mx.np if numpy else mx.nd
        if numpy:
            mx.npx.set_np()

        xs = setup(nd=nd)
        ys = compute(*xs, nd=nd)

        ys_np = [y.asnumpy() for y in ys]

        if setup_is_deterministic:
            xs = setup(nd=nd)

        xs_names = list(map(str, range(len(xs))))
        symbol_inputs = [
            mx.symbol.var(name).as_np_ndarray()
            if numpy else mx.symbol.var(name)
            for arg, name in zip(xs, xs_names)
        ]
        dc.set_variable(xs, symbol_inputs)
        with dc.context():
            ys_dc = compute(*xs, nd=nd)

        assert mode in ('all', 'symbolic', 'imperative', 'imperativewithnondccompute')
        if mode in ('all', 'imperativewithnondccompute'):
            ys_dc_np = [(y + 0).asnumpy() for y in ys_dc]
            _all_same(ys_np, ys_dc_np)

        if mode in ('all', 'imperative'):
            ys_dc_np = [y.asnumpy() for y in ys_dc]
            _all_same(ys_np, ys_dc_np)

        if mode in ('all', 'symbolic'):
            sym = dc.get_symbol(ys_dc, sym_cls=mx.sym.np._Symbol if numpy else mx.sym.Symbol)

            if setup_is_deterministic:
                xs = setup(nd=nd)

            args = {name: x for name, x in zip(xs_names, xs)}
            ys_sym = sym.bind(mx.context.current_context(), args=args).forward()

            ys_sym_np = [y.asnumpy() for y in ys_sym]
            _all_same(ys_np, ys_sym_np)
    finally:
        if numpy:
            mx.npx.reset_np()


def _all_assert_dc(setup, compute, setup_is_deterministic=True, numpy=(False, True)):
    for mode in ('all', 'symbolic', 'imperative', 'imperativewithnondccompute'):
        for numpy_ in numpy:
            _assert_dc(setup, compute, mode=mode, setup_is_deterministic=True, numpy=numpy_)


###############################################################################
# Test cases without inputs
###############################################################################
def _dc_empty_setup(*, nd):
    return []


def test_dc_no_inputs_single_output():
    def f(*, nd):
        a = nd.arange(10)
        b = a + nd.arange(a.shape[0])
        c = b - 1
        return [c]

    _all_assert_dc(_dc_empty_setup, f)


def test_dc_no_inputs_reshape():
    def f(*, nd):
        a = nd.arange(10)
        b = a + nd.arange(a.shape[0])
        c = b.reshape((5, 2))
        d = b.reshape((2, 5))
        e = (c.reshape((-1, )) + d.reshape((-1, ))) / 2
        return [c + 1, d + 1, e]

    _all_assert_dc(_dc_empty_setup, f)


def test_dc_no_inputs_slice():
    def f(*, nd):
        a = nd.arange(10)
        b = a[:5]
        if nd is mx.nd:
            c = nd.concat(b, b, dim=0)
        else:
            c = nd.concatenate([b, b], axis=0)
        return [c + a]

    _all_assert_dc(_dc_empty_setup, f)


def test_dc_no_inputs_subset_of_output():
    def f(*, nd):
        a = nd.arange(10)
        if nd is mx.nd:
            b, c = mx.nd.split(a, 2, axis=0)
        else:
            b, c = mx.np.array_split(a, 2, axis=0)
        return [b]

    _all_assert_dc(_dc_empty_setup, f)


###############################################################################
# Test cases with inputs
###############################################################################
def _dc_simple_setup(shape=(10, ), *, nd):
    n = functools.reduce(operator.mul, shape, 1)
    return [nd.arange(n).reshape(shape)]


def test_dc_single_output():
    def f(a, *, nd):
        b = a + nd.arange(a.shape[0])
        c = b - 1
        return [c]

    _all_assert_dc(_dc_simple_setup, f)


def test_dc_reshape():
    def f(a, *, nd):
        b = a + nd.arange(a.shape[0])
        c = b.reshape((5, 2))
        d = b.reshape((2, 5))
        e = (c.reshape((-1, )) + d.reshape((-1, ))) / 2
        return [c + 1, d + 1, e]

    _all_assert_dc(_dc_simple_setup, f)


def test_dc_slice():
    def f(a, *, nd):
        b = a[:5]
        if nd is mx.nd:
            c = nd.concat(b, b, dim=0)
        else:
            c = nd.concatenate([b, b], axis=0)
        return [c + a]

    _all_assert_dc(_dc_simple_setup, f)


def test_dc_subset_of_output():
    def f(a, *, nd):
        if nd is mx.nd:
            b, c = mx.nd.split(a, 2, axis=0)
        else:
            b, c = mx.np.array_split(a, 2, axis=0)
        return [b]

    _all_assert_dc(_dc_simple_setup, f)


@pytest.mark.xfail(raises=MXNetError)  # Should raise NotImplementedError https://github.com/apache/incubator-mxnet/issues/17522
def test_dc_inplace():
    def f(a, *, nd):
        a[:5] = 0
        b = a + 1
        return [a, b]

    _all_assert_dc(_dc_simple_setup, f)


###############################################################################
# Special cases
###############################################################################
def test_dc_input_part_of_output():
    a = mx.np.arange(10)
    dc.set_variable(a, mx.sym.var('a'))
    with dc.context():
        b = a + 1
    dc.get_symbol([a, b])


def test_dc_get_symbol_called_twice():
    a = mx.np.arange(10)
    dc.set_variable(a, mx.sym.var('a'))
    with dc.context():
        b = a + 1
    sym1 = dc.get_symbol(b)
    sym2 = dc.get_symbol(b)
    assert sym1.list_inputs() == ['a']
    assert sym2.list_inputs() == ['a']


@pytest.mark.xfail(raises=MXNetError)  # Should raise ValueError https://github.com/apache/incubator-mxnet/issues/17522
def test_dc_set_variable_called_twice():
    a = mx.np.arange(10)
    dc.set_variable(a, mx.sym.var('a'))
    dc.set_variable(a, mx.sym.var('b'))


def test_dc_no_inputs_context_switch():
    def f(*, nd):
        a = nd.arange(10)
        if nd is mx.nd:
            b = a.as_in_context(mx.cpu(1))
            c = (b - 1).as_in_context(mx.context.current_context())
        else:
            b = a.as_in_ctx(mx.cpu(1))
            c = (b - 1).as_in_ctx(mx.context.current_context())
        return [c]

    _assert_dc(_dc_empty_setup, f)


def test_dc_context_switch():
    def f(a, *, nd):
        if nd is mx.nd:
            b = a.as_in_context(mx.cpu(1))
            c = (b - 1).as_in_context(mx.context.current_context())
        else:
            b = a.as_in_ctx(mx.cpu(1))
            c = (b - 1).as_in_ctx(mx.context.current_context())
        return [c]

    _assert_dc(_dc_simple_setup, f)


def test_dc_astype():
    def f(a, *, nd):
        a = a.astype(np.int32)
        b = nd.zeros_like(a)
        return [a + b]

    _assert_dc(_dc_simple_setup, f)


def test_dc_dynamic_shape():
    def f(a, *, nd):
        return [mx.nd.np.flatnonzero(a)]

    # Skip GraphExecutor test due to https://github.com/apache/incubator-mxnet/issues/17810
    for mode in ('imperative', 'imperativewithnondccompute'):
        _assert_dc(_dc_simple_setup, f, mode=mode, numpy=True)


###############################################################################
# Indexing specific tests
###############################################################################
def test_dc_integer_indexing():
    def f(a, *, nd):
        return [a[1] + 1]

    _all_assert_dc(_dc_simple_setup, f)


def test_dc_slice_indexing():
    def f(a, *, nd):
        b = a.reshape((5, 2))
        return [b[:2, 1] + 1]

    _all_assert_dc(_dc_simple_setup, f)


def test_dc_tuple_indexing():
    def f(a, *, nd):
        b = a.reshape((5, 2))
        return [b[(1, 1)] + 1]

    _all_assert_dc(_dc_simple_setup, f)


def test_dc_simple_boolean_indexing():
    if mx.test_utils.default_context() == mx.gpu(0) and mx.runtime.Features().is_enabled("TVM_OP"):
        # Skip due to https://github.com/apache/incubator-mxnet/issues/17886
        return

    def setup(*, nd):
        assert nd is mx.np
        x = mx.np.array([[0, 1], [1, 1], [2, 2]])
        return [x, x < 2]

    def f(a, idx, *, nd):
        assert nd is mx.np
        return [a[idx].reshape((2, 2))]

    # Skip GraphExecutor test due to https://github.com/apache/incubator-mxnet/issues/17810
    for mode in ('imperative', 'imperativewithnondccompute'):
        _assert_dc(setup, f, mode=mode)


@pytest.mark.xfail(raises=TypeError)  # Advanced indexing
def test_dc_list_indexing():
    def f(a, *, nd):
        assert nd is mx.np
        return [a[[1, 2, 3]]]

    for mode in ('all', 'symbolic', 'imperative', 'imperativewithnondccompute'):
        _assert_dc(_dc_simple_setup, f, mode=mode)


@pytest.mark.xfail(raises=TypeError)  # Advanced indexing
def test_dc_numpy_indexing():
    def f(a, *, nd):
        assert nd is mx.np
        return [a[np.array([1, 2, 3])]]

    for mode in ('all', 'symbolic', 'imperative', 'imperativewithnondccompute'):
        _assert_dc(_dc_simple_setup, f, mode=mode)


###############################################################################
# Gluon
###############################################################################
def _assert_dc_gluon(setup, net, setup_is_deterministic=True, numpy=True, autograd=True):
    """Compare results of deferred compute and normal imperative mode.

    Parameters
    ----------
    setup : callable
        Setup function computing inputs for compute function. Always called
        outside of deferred compute.
    net : Block
    setup_is_deterministic : bool
        If True, setup function may be called multiple times. If False, will
        only be called once.
    numpy : bool
        If True, use mx.np. Otherwise mx.nd.
    autograd : bool
        Wrap in autograd

    """
    nd = mx.np if numpy else mx.nd

    xs = setup(nd=nd)
    ys = net(*xs)
    ys_np = [y.asnumpy() for y in ys]

    net.hybridize()
    if setup_is_deterministic:
        xs = setup(nd=nd)

    if autograd:
        with mx.autograd.record():
            ys_hybrid = net(*xs)
        mx.autograd.backward(ys_hybrid)
        [p.grad() for p in net.collect_params().values()]
    else:
        ys_hybrid = net(*xs)
    ys_hybrid_np = [y.asnumpy() for y in ys_hybrid]

    _all_same(ys_np, ys_hybrid_np)


def _dc_gluon_simple_setup(shape=(8, 10), *, nd):
    return [nd.ones(shape=shape, ctx=mx.context.current_context())]


def test_dc_hybridblock():
    class MyBlock(mx.gluon.HybridBlock):
        def __init__(self, *, prefix=None, params=None):
            super().__init__(prefix, params)
            with self.name_scope():
                self.dense = mx.gluon.nn.Dense(units=10, in_units=10)
                self.weight = self.params.get('weight', shape=(10, ))

        def forward(self, x):
            assert x.shape[1] == 10  # due to in_units=10 above
            return self.dense(x) + self.weight.data(x.context)

    net = MyBlock()
    net.initialize()
    _assert_dc_gluon(_dc_gluon_simple_setup, net, numpy=False)
    with mx.util.np_array(True):
        net = MyBlock()
        net.initialize()
        _assert_dc_gluon(_dc_gluon_simple_setup, net, numpy=True)


@pytest.mark.xfail(raises=RuntimeError)
def test_dc_hybridblock_deferred_init_no_infer_shape():
    class MyBlock(mx.gluon.HybridBlock):
        def __init__(self, *, prefix=None, params=None):
            super().__init__(prefix, params)
            with self.name_scope():
                self.dense = mx.gluon.nn.Dense(units=10)
                self.weight = self.params.get('weight', allow_deferred_init=True)

        def forward(self, x):
            return self.dense(x) + self.weight.data(x.context)

    net = MyBlock()
    net.initialize()
    data = mx.nd.ones(shape=(8, 10), ctx=mx.context.current_context())
    net(data)  # Raises RuntimeError


def test_dc_hybridblock_deferred_init():
    class MyBlock(mx.gluon.HybridBlock):
        def __init__(self, *, prefix=None, params=None):
            super().__init__(prefix, params)
            with self.name_scope():
                self.dense = mx.gluon.nn.Dense(units=10)
                self.weight = self.params.get('weight', allow_deferred_init=True)

        def infer_shape(self, x):
            self.weight.shape = (x.shape[1], )

        def forward(self, x):
            return self.dense(x) + self.weight.data(x.context)

    net = MyBlock()
    net.initialize()
    _assert_dc_gluon(_dc_gluon_simple_setup, net, numpy=False)
    with mx.util.np_array(True):
        net = MyBlock()
        net.initialize()
        _assert_dc_gluon(_dc_gluon_simple_setup, net, numpy=True)


def test_dc_hybridblock_dynamic_shape():
    if mx.test_utils.default_context() == mx.gpu(0) and mx.runtime.Features().is_enabled("TVM_OP"):
        # Skip due to https://github.com/apache/incubator-mxnet/issues/17886
        return

    class MyBlock(mx.gluon.HybridBlock):
        def __init__(self, *, prefix=None, params=None):
            super().__init__(prefix, params)
            with self.name_scope():
                self.dense = mx.gluon.nn.Dense(units=10)

        def forward(self, x, idx):
            return x[idx].reshape((2, 2)), mx.np.flatnonzero(self.dense(x))

    def setup(*, nd):
        assert nd is mx.np
        x = mx.np.array([[0, 1], [1, 1], [2, 2]])
        return [x, x < 2]

    with mx.util.np_array(True):
        net = MyBlock()
        net.initialize()
        _assert_dc_gluon(setup, net, numpy=True)


@pytest.mark.xfail(raises=RuntimeError)
def test_dc_hybridblock_symbolblock():
    model = mx.gluon.nn.HybridSequential()
    model.add(mx.gluon.nn.Dense(128, activation='tanh'))
    model.add(mx.gluon.nn.Dropout(0.5))
    model.add(mx.gluon.nn.Dense(64, activation='tanh'),
              mx.gluon.nn.Dense(32, in_units=64))
    model.add(mx.gluon.nn.Activation('relu'))
    model.initialize()
    inputs = mx.sym.var('data')
    outputs = model(inputs).get_internals()
    smodel = mx.gluon.SymbolBlock(outputs, inputs, params=model.collect_params())
    assert len(smodel(mx.nd.zeros((16, 10)))) == 14

    class Net(mx.gluon.HybridBlock):
        def __init__(self, model):
            super(Net, self).__init__()
            self.model = model

        def forward(self, x):
            out = self.model(x)
            return mx.nd.add_n(*[i.sum() for i in out])

    net = Net(smodel)
    net.hybridize()
    data = mx.nd.zeros((16, 10))
    out = net(data)
    out.asnumpy()

    net.hybridize()
    out_hybrid = net(data)  # Raises RuntimeError

    _all_same([out], [out_hybrid])

