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

import copy
import numpy as np
import mxnet as mx
from mxnet import gluon
from mxnet.test_utils import *
from mxnet.base import _as_list
from collections import defaultdict
from mxnet.attribute import AttrScope

@mx.util.use_np
def test_while_loop_simple_forward():

    class _TestBlock(gluon.HybridBlock):

        def __init__(self, cond, func, max_iterations):
            super(_TestBlock, self).__init__()
            self.cond = cond
            self.func = func
            self.max_iterations = max_iterations

        def forward(self, *loop_vars):
            return mx.npx.while_loop(
                cond=self.cond,
                func=self.func,
                loop_vars=loop_vars,
                max_iterations=self.max_iterations
            )

    for hybridize in [False, True]:
        # Case 1.1: result should be sum([1, 2, 3 ... 100])
        model = _TestBlock(
            cond=lambda i, s: i <= 5,
            func=lambda i, s: (None, (i + 1, s + i)),
            max_iterations=10,
        )
        if hybridize:
            model.hybridize()
        _, result = model(
            mx.np.array([1], dtype="int64"), # i
            mx.np.array([0], dtype="int64"), # s
        )
        assert result[0].item() == 6
        assert result[1].item() == 15
        # Case 1.2: result should be sum([1, 2, 3 ... 1000])
        model = _TestBlock(
            cond=lambda i, s, true: true,
            func=lambda i, s, true: (None, (i + 1, s + i, true)),
            max_iterations=1000,
        )
        if hybridize:
            model.hybridize()
        _, result = model(
            mx.np.array([1], dtype="int64"), # i
            mx.np.array([0], dtype="int64"), # s
            mx.np.array([1], dtype="int64"), # true
        )
        assert result[0].item() == 1001
        assert result[1].item() == 500500
        assert result[2].item() == 1
        # Case 1.3: result should be sum([])
        model = _TestBlock(
            cond=lambda i, s, false: false,
            func=lambda i, s, false: (None, (i + 1, s + i, false)),
            max_iterations=1000,
        )
        if hybridize:
            model.hybridize()
        _, result = model(
            mx.np.array([1], dtype="int64"), # i
            mx.np.array([0], dtype="int64"), # s
            mx.np.array([0], dtype="int64"), # false
        )
        assert result[0].item() == 1
        assert result[1].item() == 0
        assert result[2].item() == 0
        # Case 2.1: result should be sum([1, 2, 3 ... 100])
        model = _TestBlock(
            cond=lambda i, s: i <= 100,
            func=lambda i, s: (i, (i + 1, s + i)),
            max_iterations=1000,
        )
        if hybridize:
            model.hybridize()
        outputs, (result_i, result_s) = model(
            mx.np.array([1], dtype="int64"), # i
            mx.np.array([0], dtype="int64"), # s
        )
        assert all(outputs.asnumpy()[ : 100] == np.arange(1, 101).reshape(100, 1))
        assert result_i.item() == 101
        assert result_s.item() == 5050
        # Case 2.2: result should be sum([1, 2, 3 ... 1000])
        model = _TestBlock(
            cond=lambda i, s, true: true,
            func=lambda i, s, true: (i, (i + 1, s + i, true)),
            max_iterations=1000,
        )
        if hybridize:
            model.hybridize()
        outputs, (result_i, result_s, _) = model(
            mx.np.array([1], dtype="int64"), # i
            mx.np.array([0], dtype="int64"), # s
            mx.np.array([1], dtype="int64"), # true
        )
        assert all(outputs.asnumpy() == np.arange(1, 1001).reshape(1000, 1))
        assert result_i.item() == 1001
        assert result_s.item() == 500500
        # Case 2.3: a corner case, in which loop body is never executed
        model = _TestBlock(
            cond=lambda i, s, false: false,
            func=lambda i, s, false: (i, (i + 1, s + i, false)),
            max_iterations=1000,
        )
        if hybridize:
            model.hybridize()
        _, (result_i, result_s, _) = model(
            mx.np.array([1], dtype="int64"), # i
            mx.np.array([0], dtype="int64"), # s
            mx.np.array([0], dtype="int64"), # false
        )
        assert result_i.item() == 1
        assert result_s.item() == 0


def test_cut_subgraph_foreach():
    class TestLayer(gluon.HybridBlock):
        def __init__(self):
            super(TestLayer, self).__init__()

        def forward(self, inputs, states):
            def step1(data, states):
                return data + 1, states
            out1, states1 = mx.npx.foreach(step1, inputs, states)
            out2, states2 = mx.npx.foreach(step1, out1, states)
            def step2(data, states):
                return data + states[0], states
            out, states = mx.npx.foreach(step2, out2, states1)
            return out

    data = mx.np.random.normal(loc=0, scale=1, size=(5, 10))
    states = mx.np.random.normal(loc=0, scale=1, size=(10))
    layer = TestLayer()
    layer.initialize(ctx=default_context())
    res1 = layer(data, [states])

    with mx.autograd.record():
        res1 = layer(data, [states])

    layer = TestLayer()
    layer.initialize(ctx=default_context())
    layer.hybridize()
    res2 = layer(data, [states])

    with mx.autograd.record():
        res2 = layer(data, [states])
    assert_almost_equal(res1.asnumpy(), res2.asnumpy(), rtol=1e-3, atol=1e-3)


@mx.util.use_np
def test_uniq_name():
    class ForeachLayer1(gluon.HybridBlock):
        def __init__(self):
            super(ForeachLayer1, self).__init__()

        def forward(self, inputs, states):
            def step1(data, states):
                return data + 1, states
            out1, states1 = mx.npx.foreach(step1, inputs, states)
            # The input variables have the same symbol name.
            out, states = mx.npx.foreach(step1, out1, states1)
            return out

    class ForeachLayer2(gluon.HybridBlock):
        def __init__(self):
            super(ForeachLayer2, self).__init__()

        def forward(self, inputs, states):
            def step1(data, states):
                return data + 1, states
            out1, states1 = mx.npx.foreach(step1, inputs, states)
            def step2(data, states):
                return data, [states[0] + states[0] + mx.np.squeeze(mx.npx.slice(data, begin=0, end=1))]
            # The input variables have the same symbol names.
            # The free variables have the same symbol names as the input variables.
            out, states = mx.npx.foreach(step2, out1, states1)
            return out

    class WhileLayer1(gluon.HybridBlock):
        def __init__(self):
            super(WhileLayer1, self).__init__()

        def forward(self, inputs, states):
            def cond(state1, state2):
                s = mx.np.squeeze(mx.npx.slice(state1, begin=0, end=1))
                return s == s
            def step(state1, state2):
                return state1 + 1, [state1 + 1, state2 + 1]
            states = [states[0], states[0] + 1]
            out1, states1 = mx.npx.while_loop(cond, step, states, max_iterations=5)
            # The input variables have the same symbol name.
            out, states = mx.npx.while_loop(cond, step, states1, max_iterations=5)
            return out

    class WhileLayer2(gluon.HybridBlock):
        def __init__(self):
            super(WhileLayer2, self).__init__()

        def forward(self, inputs, states):
            def cond(state1, state2):
                s = mx.np.squeeze(mx.npx.slice(state1, begin=0, end=1))
                return s == s
            def step1(state1, state2):
                return state1 + 1, [state1, state2]
            states = [states[0], states[0] + 1]
            out1, states1 = mx.npx.while_loop(cond, step1, states, max_iterations=5)
            def step2(state1, state2):
                return state1 + 1, [state1 + state1[0], state2 + state1[1]]
            # The input variables have the same symbol name.
            out, states = mx.npx.while_loop(cond, step2, states1, max_iterations=5)
            return out

    TestLayers = [ForeachLayer1, ForeachLayer2,
            WhileLayer1, WhileLayer2]
    # TestLayers = [WhileLayer1]

    data = mx.np.random.normal(loc=0, scale=1, size=(2, 5))
    states = mx.np.random.normal(loc=0, scale=1, size=(5))
    for TestLayer in TestLayers:
        layer = TestLayer()
        layer.initialize(ctx=default_context())
        res1 = layer(data, [states])

        with mx.autograd.record():
            res1 = layer(data, [states])

        layer = TestLayer()
        layer.initialize(ctx=default_context())
        layer.hybridize()
        res2 = layer(data, [states])

        with mx.autograd.record():
            res2 = layer(data, [states])
        assert_almost_equal(res1.asnumpy(), res2.asnumpy(), rtol=0.001, atol=0.0001)


@mx.util.use_np
def test_cut_subgraph_while_loop():
    class TestLayer(gluon.HybridBlock):
        def __init__(self):
            super(TestLayer, self).__init__()
        def forward(self, data):
            out1, data1 = mx.npx.while_loop(
                cond=lambda i: i <= 5,
                func=lambda i: (None, (i + 1, )),
                loop_vars=(data, ),
                max_iterations=10,
            )
            out2, data2 = mx.npx.while_loop(
                cond=lambda i: i,
                func=lambda i: (None, (i + 1, )),
                loop_vars=data1[0],
                max_iterations=10,
            )
            return data2[0]
    data = mx.np.random.normal(loc=0, scale=1, size=(1, ))
    layer = TestLayer()
    layer.initialize(ctx=default_context())
    res1 = layer(data)
    with mx.autograd.record():
        res1 = layer(data)
    layer = TestLayer()
    layer.initialize(ctx=default_context())
    layer.hybridize()
    res2 = layer(data)
    with mx.autograd.record():
        res2 = layer(data)
    assert_almost_equal(res1.asnumpy(), res2.asnumpy(), rtol=1e-3, atol=1e-3)


@mx.util.use_np
def test_cut_subgraph_cond():
    class TestLayer(gluon.HybridBlock):
        def __init__(self):
            super(TestLayer, self).__init__()
        def forward(self, data):
            data1 = mx.npx.cond(
                pred=lambda data: data > 0.5,
                then_func=lambda data: data * 2,
                else_func=lambda data: data * 3,
                inputs=data,
            )
            data2 = mx.npx.cond(
                pred=lambda data: data > 0.5,
                then_func=lambda data: data * 2,
                else_func=lambda data: data * 3,
                inputs=data1,
            )
            return data2
    data = mx.np.random.normal(loc=0, scale=1, size=(1, ))
    layer = TestLayer()
    layer.initialize(ctx=default_context())
    res1 = layer(data)
    with mx.autograd.record():
        res1 = layer(data)
    layer = TestLayer()
    layer.initialize(ctx=default_context())
    layer.hybridize()
    res2 = layer(data)
    with mx.autograd.record():
        res2 = layer(data)
    assert_almost_equal(res1.asnumpy(), res2.asnumpy(), rtol=1e-3, atol=1e-3)


@mx.util.use_np
def test_output_format_foreach():
    class TestLayer1(gluon.HybridBlock):
        def __init__(self, step):
            super(TestLayer1, self).__init__()
            self.step = step
        def forward(self, ins, states):
            out, states = mx.npx.foreach(self.step, ins, states)
            return out, states

    def step1(data, state):
        return data, state
    def step2(data, state):
        return [data], state
    def step3(data, state):
        if isinstance(state, list):
            return [], [state[0] + data]
        else:
            return [], state + data
    def step4(data, state):
        if isinstance(state, list):
            return [data, state[0]], state
        else:
            return [data, state], state

    steps = [step1, step2, step3, step4]
    data = mx.np.random.normal(loc=0, scale=1, size=(10, 2))
    state = mx.np.random.normal(loc=0, scale=1, size=(2))
    for step in steps:
        layer1 = TestLayer1(step)
        layer1.initialize(ctx=default_context())
        layer2 = TestLayer1(step)
        layer2.initialize(ctx=default_context())
        layer2.hybridize()
        out1, state1 = layer1(data, [state])
        out2, state2 = layer2(data, [state])
        step_out, step_state = step(data, [state])
        assert type(out1) == type(step_out)
        assert type(out2) == type(step_out)
        assert type(state1) == type(step_state)
        assert type(state2) == type(step_state)
        out1 = _as_list(out1)
        out2 = _as_list(out2)
        state1 = _as_list(state1)
        state2 = _as_list(state2)
        for i in range(len(out1)):
            assert_almost_equal(out1[i].asnumpy(), out2[i].asnumpy(), rtol=0.001, atol=0.0001)
        for i in range(len(state1)):
            assert_almost_equal(state1[i].asnumpy(), state2[i].asnumpy(), rtol=0.001, atol=0.0001)

        layer1 = TestLayer1(step)
        layer1.initialize(ctx=default_context())
        layer2 = TestLayer1(step)
        layer2.initialize(ctx=default_context())
        layer2.hybridize()
        out1, state1 = layer1(data, state)
        out2, state2 = layer2(data, state)
        step_out, step_state = step(data, state)
        assert type(out1) == type(step_out)
        assert type(out2) == type(step_out)
        assert type(state1) == type(step_state)
        assert type(state2) == type(step_state)
        out1 = _as_list(out1)
        out2 = _as_list(out2)
        state1 = _as_list(state1)
        state2 = _as_list(state2)
        for i in range(len(out1)):
            assert_almost_equal(out1[i].asnumpy(), out2[i].asnumpy(), rtol=0.001, atol=0.0001)
        for i in range(len(state1)):
            assert_almost_equal(state1[i].asnumpy(), state2[i].asnumpy(), rtol=0.001, atol=0.0001)

        if step == step3:
            continue
        layer1 = TestLayer1(step)
        layer1.initialize(ctx=default_context())
        layer2 = TestLayer1(step)
        layer2.initialize(ctx=default_context())
        layer2.hybridize()
        out1, state1 = layer1(data, [state, [state + 1]])
        out2, state2 = layer2(data, [state, [state + 1]])
        step_out, step_state = step(data, [state, [state + 1]])
        assert type(out1) == type(step_out)
        assert type(out2) == type(step_out)
        assert type(state1) == type(step_state)
        assert type(state2) == type(step_state)
        out1 = _as_list(out1)
        out2 = _as_list(out2)
        state1 = _as_list(state1)
        state2 = _as_list(state2)
        for i in range(len(out1)):
            assert_almost_equal(out1[i].asnumpy(), out2[i].asnumpy(), rtol=0.001, atol=0.0001)
        for i in range(len(state1)):
            if isinstance(state1[i], list):
                assert_almost_equal(state1[i][0].asnumpy(), state2[i][0].asnumpy(),
                        rtol=0.001, atol=0.0001)
            else:
                assert_almost_equal(state1[i].asnumpy(), state2[i].asnumpy(),
                        rtol=0.001, atol=0.0001)


@mx.util.use_np
def test_output_format_while():
    class TestLayer1(gluon.HybridBlock):
        def __init__(self, step, use_list, nested_list=False):
            super(TestLayer1, self).__init__()
            self.step = step
            self.use_list = use_list
            self.nested_list = nested_list
        def forward(self, states):
            def cond(state1):
                scalar = mx.npx.slice(state1, begin=0, end=1)
                return scalar == scalar
            cond_func = cond
            if self.use_list:
                states = [states]
            elif self.nested_list:
                def cond2(state1, state2):
                    scalar = mx.npx.slice(state1, begin=0, end=1)
                    return scalar == scalar
                cond_func = cond2
                states = [states, [states + 1]]
            out, states = mx.npx.while_loop(cond_func, self.step, states, max_iterations=5)
            return out, states

    def step1(state):
        return state, state
    def step2(state):
        if isinstance(state, list):
            return state, state
        else:
            return [state], state
    def step3(state):
        return [], state

    steps = [step1, step2, step3]
    state = mx.np.random.normal(loc=0, scale=1, size=(2))
    for step in steps:
        layer1 = TestLayer1(step, False)
        layer1.initialize(ctx=default_context())
        layer2 = TestLayer1(step, False)
        layer2.initialize(ctx=default_context())
        layer2.hybridize()
        out1, state1 = layer1(state)
        out2, state2 = layer2(state)
        assert type(out1) == type(out2)
        assert type(state1) == type(state1)
        out1 = _as_list(out1)
        out2 = _as_list(out2)
        state1 = _as_list(state1)
        state2 = _as_list(state2)
        for i in range(len(out1)):
            assert_almost_equal(out1[i].asnumpy(), out2[i].asnumpy(), rtol=0.001, atol=0.0001)
        for i in range(len(state1)):
            assert_almost_equal(state1[i].asnumpy(), state2[i].asnumpy(), rtol=0.001, atol=0.0001)

        layer1 = TestLayer1(step, True)
        layer1.initialize(ctx=default_context())
        layer2 = TestLayer1(step, True)
        layer2.initialize(ctx=default_context())
        layer2.hybridize()
        out1, state1 = layer1(state)
        out2, state2 = layer2(state)
        assert type(out1) == type(out2)
        assert type(state1) == type(state2)
        out1 = _as_list(out1)
        out2 = _as_list(out2)
        state1 = _as_list(state1)
        state2 = _as_list(state2)
        for i in range(len(out1)):
            assert_almost_equal(out1[i].asnumpy(), out2[i].asnumpy(), rtol=0.001, atol=0.0001)
        for i in range(len(state1)):
            assert_almost_equal(state1[i].asnumpy(), state2[i].asnumpy(), rtol=0.001, atol=0.0001)

    def step4(state, state2):
        states = _as_list(state)
        states.append(state2)
        return state, states
    def step5(state, state2):
        states = _as_list(state)
        states.append(state2)
        if isinstance(state, list):
            return state, states
        else:
            return [state], states
    def step6(state, state2):
        states = _as_list(state)
        states.append(state2)
        return [], states

    steps = [step4, step5, step6]
    for step in steps:
        layer1 = TestLayer1(step, False, True)
        layer1.initialize(ctx=default_context())
        layer2 = TestLayer1(step, False, True)
        layer2.initialize(ctx=default_context())
        layer2.hybridize()
        out1, state1 = layer1(state)
        out2, state2 = layer2(state)
        assert type(out1) == type(out2)
        assert type(state1) == type(state2)
        out1 = _as_list(out1)
        out2 = _as_list(out2)
        state1 = _as_list(state1)
        state2 = _as_list(state2)
        for i in range(len(out1)):
            assert_almost_equal(out1[i].asnumpy(), out2[i].asnumpy(), rtol=0.001, atol=0.0001)
        for i in range(len(state1)):
            if not isinstance(state1[i], list):
                assert_almost_equal(state1[i].asnumpy(), state2[i].asnumpy(),
                                    rtol=0.001, atol=0.0001)


@mx.util.use_np
def test_output_format_cond():
    class TestLayer1(gluon.HybridBlock):
        def __init__(self, func):
            super(TestLayer1, self).__init__()
            self.func = func
        def forward(self, data):
            def then_func(data):
                return self.func(data)
            def else_func(data):
                return self.func(data)
            return mx.npx.cond(lambda data: mx.npx.slice(data, begin=0, end=1),
                    then_func, else_func, data)

    def func1(data):
        return data
    def func2(data):
        return [data]
    def func3(data):
        return [data, data]

    funcs = [func1, func2, func3]
    data = mx.np.random.normal(loc=0, scale=1, size=(2))
    for func in funcs:
        layer1 = TestLayer1(func)
        layer1.initialize(ctx=default_context())
        layer2 = TestLayer1(func)
        layer2.initialize(ctx=default_context())
        layer2.hybridize()
        out1 = layer1(data)
        out2 = layer2(data)
        func_out = func(data)
        assert type(out1) == type(func_out)
        assert type(out2) == type(func_out)
        out1 = _as_list(out1)
        out2 = _as_list(out2)
        for i in range(len(out1)):
            assert_almost_equal(out1[i].asnumpy(), out2[i].asnumpy(), rtol=0.001, atol=0.0001)


@mx.util.use_np
def test_scope():
    class TestBlock1(gluon.HybridBlock):
        def __init__(self):
            super(TestBlock1, self).__init__()

        def forward(self, data):
            (new_data, ) = mx.npx.cond(
                pred=lambda data: data > 0.5,
                then_func=lambda data: data * 2,
                else_func=lambda data: data * 3,
                inputs=data,
                name="my_cond",
            )
            return new_data

    class TestBlock2(gluon.HybridBlock):
        def __init__(self):
            super(TestBlock2, self).__init__()

        def forward(self, data):
            (new_data, ) = mx.npx.cond(
                pred=lambda data: data > 0.5,
                then_func=lambda data: data * 2,
                else_func=lambda data: data * 3,
                inputs=data,
                name="my_cond",
            )
            return new_data

    AttrScope._subgraph_names = defaultdict(int)
    data = mx.np.random.normal(loc=0, scale=1, size=(1, ))
    with AttrScope(__subgraph_name__="my_cond"):
        block1 = TestBlock1()
        block1.initialize(ctx=default_context())
        block1.hybridize()
        _ = block1(data)
        block2 = TestBlock2()
        block2.initialize(ctx=default_context())
        block2.hybridize()
        _ = block2(data)
        assert len(AttrScope._subgraph_names) == 3
        assert AttrScope._subgraph_names['my_cond$my_cond_else'] == 2
        assert AttrScope._subgraph_names['my_cond$my_cond_pred'] == 2
        assert AttrScope._subgraph_names['my_cond$my_cond_then'] == 2


class RNNLayer(gluon.HybridBlock):
    def __init__(self, cell_type, hidden_size):
        super(RNNLayer, self).__init__()
        self.cell = cell_type(hidden_size)

    def forward(self, inputs, states):
        out, states = mx.npx.foreach(self.cell, inputs, states)
        return out
    
    def infer_shape(self, input, *args):
        self.cell.infer_shape(0, input, False)

@mx.util.use_np
def check_rnn(cell_type, num_states):
    batch_size = 10
    hidden_size = 100
    rnn_data = mx.np.random.normal(loc=0, scale=1, size=(5, batch_size, 50))
    state_shape = (batch_size, hidden_size)
    states = [mx.np.random.normal(loc=0, scale=1, size=state_shape) for i in range(num_states)]
    layer = RNNLayer(cell_type, hidden_size)
    layer.infer_shape(rnn_data)
    layer.initialize(ctx=default_context())
    res1 = layer(rnn_data, states)
    params1 = layer.collect_params()
    orig_params1 = copy.deepcopy(params1)

    trainer = gluon.Trainer(params1, 'sgd', {'learning_rate' : 0.03})
    with mx.autograd.record():
        res1 = layer(rnn_data, states)
    res1.backward()
    trainer.step(batch_size)

    configs = [
            {},
            {'inline_limit': 0},
            {'static_alloc': True},
            {'static_alloc': True, 'static_shape': True} ]
    for config in configs:
        layer = RNNLayer(cell_type, hidden_size)
        layer.infer_shape(rnn_data)
        layer.initialize(ctx=default_context())
        layer.hybridize(**config)
        res2 = layer(rnn_data, states)
        params2 = layer.collect_params()
        for key, val in orig_params1.items():
            params2[key].set_data(copy.deepcopy(val.data()))
        trainer = gluon.Trainer(params2, 'sgd', {'learning_rate' : 0.03})
        with mx.autograd.record():
            res2 = layer(rnn_data, states)
        assert_almost_equal(res1.asnumpy(), res2.asnumpy(), rtol=1e-3, atol=1e-3)
        res2.backward()
        trainer.step(batch_size)

        for key, val in params1.items():
            weight1 = val.data()
            weight2 = params2[key].data()
            assert_almost_equal(weight1.asnumpy(), weight2.asnumpy(),
                    rtol=1e-3, atol=1e-3)


def test_rnn():
    cell_types = [(gluon.rnn.RNNCell, 1), (gluon.rnn.LSTMCell, 2),
            (gluon.rnn.GRUCell, 1)]
    for cell_type, num_states in cell_types:
        check_rnn(cell_type, num_states)

