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

import mxnet as mx
from mxnet import gluon
import numpy as np
import copy
from numpy.testing import assert_allclose
import unittest
from mxnet.test_utils import almost_equal, default_context
from numpy.testing import assert_allclose as assert_almost_equal  # This is more restrictive
from mxnet.base import _as_list


def test_while_loop_simple_forward():

    class _TestBlock(gluon.HybridBlock):

        def __init__(self, cond, func, max_iterations):
            super(_TestBlock, self).__init__()
            self.cond = cond
            self.func = func
            self.max_iterations = max_iterations

        def hybrid_forward(self, F, *loop_vars):
            return F.contrib.while_loop(
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
            model.hybridize(inline_limit=0)
        _, result = model(
            mx.nd.array([1], dtype="int64"), # i
            mx.nd.array([0], dtype="int64"), # s
        )
        assert result[0].asscalar() == 6
        assert result[1].asscalar() == 15
        # Case 1.2: result should be sum([1, 2, 3 ... 1000])
        model = _TestBlock(
            cond=lambda i, s, true: true,
            func=lambda i, s, true: (None, (i + 1, s + i, true)),
            max_iterations=1000,
        )
        if hybridize:
            model.hybridize(inline_limit=0)
        _, result = model(
            mx.nd.array([1], dtype="int64"), # i
            mx.nd.array([0], dtype="int64"), # s
            mx.nd.array([1], dtype="int64"), # true
        )
        assert result[0].asscalar() == 1001
        assert result[1].asscalar() == 500500
        assert result[2].asscalar() == 1
        # Case 1.3: result should be sum([])
        model = _TestBlock(
            cond=lambda i, s, false: false,
            func=lambda i, s, false: (None, (i + 1, s + i, false)),
            max_iterations=1000,
        )
        if hybridize:
            model.hybridize(inline_limit=0)
        _, result = model(
            mx.nd.array([1], dtype="int64"), # i
            mx.nd.array([0], dtype="int64"), # s
            mx.nd.array([0], dtype="int64"), # false
        )
        assert result[0].asscalar() == 1
        assert result[1].asscalar() == 0
        assert result[2].asscalar() == 0
        # Case 2.1: result should be sum([1, 2, 3 ... 100])
        model = _TestBlock(
            cond=lambda i, s: i <= 100,
            func=lambda i, s: (i, (i + 1, s + i)),
            max_iterations=1000,
        )
        if hybridize:
            model.hybridize(inline_limit=0)
        (outputs, ), (result_i, result_s) = model(
            mx.nd.array([1], dtype="int64"), # i
            mx.nd.array([0], dtype="int64"), # s
        )
        assert all(outputs.asnumpy()[ : 100] == np.arange(1, 101).reshape(100, 1))
        assert result_i.asscalar() == 101
        assert result_s.asscalar() == 5050
        # Case 2.2: result should be sum([1, 2, 3 ... 1000])
        model = _TestBlock(
            cond=lambda i, s, true: true,
            func=lambda i, s, true: (i, (i + 1, s + i, true)),
            max_iterations=1000,
        )
        if hybridize:
            model.hybridize(inline_limit=0)
        (outputs, ), (result_i, result_s, _) = model(
            mx.nd.array([1], dtype="int64"), # i
            mx.nd.array([0], dtype="int64"), # s
            mx.nd.array([1], dtype="int64"), # true
        )
        assert all(outputs.asnumpy() == np.arange(1, 1001).reshape(1000, 1))
        assert result_i.asscalar() == 1001
        assert result_s.asscalar() == 500500
        # Case 2.3: very corner case
        model = _TestBlock(
            cond=lambda i, s, false: false,
            func=lambda i, s, false: (i, (i + 1, s + i, false)),
            max_iterations=1000,
        )
        if hybridize:
            model.hybridize(inline_limit=0)
        _, (result_i, result_s, _) = model(
            mx.nd.array([1], dtype="int64"), # i
            mx.nd.array([0], dtype="int64"), # s
            mx.nd.array([0], dtype="int64"), # false
        )
        assert result_i.asscalar() == 1
        assert result_s.asscalar() == 0


def _verify_while_loop(cond, func, loop_var_shapes, free_var_shapes, is_train, max_iterations, is_for):

    def _create_vars(num, prefix):
        return [mx.sym.var(prefix + str(i)) for i in range(num)]

    def _create_arrays(shapes):
        return [mx.nd.random.uniform(-1.0, 1.0, shape=x) for x in shapes]

    def _create_dict(prefix, shapes):
        return {prefix + str(i): mx.nd.random.uniform(-1.0, 1.0, shape=x) for i, x in enumerate(shapes)}

    def _merge_dict(*dicts):
        result = {}
        for item in dicts:
            result.update(item)
        return result

    def _to_numpy_list(arrays):
        return [x.asnumpy() if x is not None else x for x in arrays]

    def _get_imperative_result():
        free_vars = [args["FreeVar" + str(i)].copy() for i, _ in enumerate(free_var_shapes)]
        loop_vars = [args["LoopVar" + str(i)].copy() for i, _ in enumerate(loop_var_shapes)]
        loop_var_start = int(is_for)
        if is_train:
            for var in free_vars + loop_vars[loop_var_start: ]:
                var.attach_grad()
        with mx.autograd.record(train_mode=is_train):
            outputs, final_loop_vars = mx.nd.contrib.while_loop(
                cond=lambda *_loop_vars: cond(_loop_vars, free_vars),
                func=lambda *_loop_vars: func(_loop_vars, free_vars),
                loop_vars=loop_vars,
                max_iterations=max_iterations,
            )
            n_steps = outputs[0].shape[0] if outputs else 0
            out_grads = _create_arrays(x.shape for x in outputs)  \
                      + _create_arrays(x.shape for x in final_loop_vars)
            loop_result_nd = [x * 2 for x in outputs] + [x * 3 for x in final_loop_vars]
            grads = []
            if is_train:
                cat_out = mx.nd.concat(*[x.reshape(-1) for x in loop_result_nd], dim=0)
                cat_out.backward(out_grad=mx.nd.concat(*[x.reshape(-1) for x in out_grads], dim=0))
                grads = [free_vars[i].grad for i, _ in enumerate(free_var_shapes)] \
                      + [loop_vars[i].grad for i, _ in enumerate(loop_var_shapes) if i >= loop_var_start]
            return _to_numpy_list(loop_result_nd), _to_numpy_list(grads), out_grads, n_steps

    def _get_symbolic_result(out_grads, n_steps):

        def _copy_args_dict(name_list):
            return {name: args[name].copy() for name in name_list}

        def _zeros_like_dict(name_list):
            return {name: mx.nd.zeros_like(args[name]) for name in name_list}

        free_syms = _create_vars(len(free_var_shapes), "FreeVar")
        loop_syms = _create_vars(len(loop_var_shapes), "LoopVar")
        outputs, final_loop_syms = mx.sym.contrib.while_loop(
            cond=lambda *_loop_vars: cond(_loop_vars, free_syms),
            func=lambda *_loop_vars: func(_loop_vars, free_syms),
            loop_vars=loop_syms,
            max_iterations=max_iterations,
        )
        if n_steps == 0:
            outputs = []
        else:
            outputs = [x.slice_axis(axis=0, begin=0, end=n_steps) for x in outputs]
        loop_result_sym = [x * 2 for x in outputs] + [x * 3 for x in final_loop_syms]
        loop_result_sym = mx.sym.Group(loop_result_sym)

        loop_var_start = int(is_for)
        args_names = ["FreeVar" + str(i) for i, _ in enumerate(free_var_shapes)] \
                   + ["LoopVar" + str(i) for i, _ in enumerate(loop_var_shapes) if i >= loop_var_start]
        args_grad = None if not is_train else _zeros_like_dict(x for x in args_names)
        executor = loop_result_sym.bind(
            ctx=default_context(),
            args=_copy_args_dict(loop_result_sym.list_inputs()),
            args_grad=args_grad,
        )
        loop_result_nd = executor.forward(is_train=is_train)
        grads = []
        if is_train:
            executor.backward(out_grads=out_grads)
            grads = [executor.grad_dict.get("FreeVar" + str(i), None) for i, _ in enumerate(free_var_shapes)] \
                  + [executor.grad_dict.get("LoopVar" + str(i), None) for i, _ in enumerate(loop_var_shapes) if i >= loop_var_start]
        return _to_numpy_list(loop_result_nd), _to_numpy_list(grads)

    args = _merge_dict(
        _create_dict("FreeVar", free_var_shapes),
        _create_dict("LoopVar", loop_var_shapes),
    )
    if is_for:
        assert loop_var_shapes[0] == (1, )
        args["LoopVar0"] = mx.nd.array([0])
    imp_outs, imp_grads, out_grads, n_steps = _get_imperative_result()
    sym_outs, sym_grads = _get_symbolic_result(out_grads, n_steps)
    for imp_out, sym_out in zip(imp_outs, sym_outs):
        if imp_out is None or sym_out is None:
            continue
        assert_almost_equal(imp_out, sym_out)
    for imp_grad, sym_grad in zip(imp_grads, sym_grads):
        if imp_grad is None or sym_grad is None:
            continue
        assert_almost_equal(imp_grad, sym_grad, rtol=1e-5, atol=1e-5)


def test_while_loop_for_foreach():

    def make_true_cond():
        return lambda loop_vars, _: (loop_vars[0] < 1e9).prod()

    def make_false_cond():
        return lambda loop_vars, _: (loop_vars[0] > 1e9).prod()

    def make_for_cond(length):
        return lambda loop_vars, _: loop_vars[0] < length

    def case_0():
        # This is a simple testcase that all loop steps are independent'
        # It basically scans the array and outputs itself
        # There is 1 output
        # There is 1 state: i
        def _simple_func(loop, free):
            (i, ), (scanned, ) = loop, free
            in_ = scanned.take(i).squeeze(axis=0)
            return (in_, i + 1)
        _verify_while_loop(
            cond=make_true_cond(),
            func=_simple_func,
            max_iterations=1,
            is_train=True,
            is_for=True,
            loop_var_shapes=[
                (1, ),          # i
            ],
            free_var_shapes=[
                (1, 3),         # scanned
            ],
        )

    def case_1(**params):
        # This is a simple testcase that simulates a cumulative sum
        # There is 1 output
        # There is 1 state: s
        step_funcs = [
            lambda a, b, s: s,
            lambda a, b, s: a * 1.5 + b * 2.5 - s * 3.5,
            lambda a, b, s: a * 1.5 - s * 3.5 + b * 2.5,
            lambda a, b, s: b * 2.5 + a * 1.5 - s * 3.5,
            lambda a, b, s: b * 2.5 - s * 3.5 + a * 1.5,
            lambda a, b, s: s * -3.5 + a * 1.5 + b * 2.5,
            lambda a, b, s: s * -3.5 + b * 2.5 + a * 1.5,
            lambda a, b, s: a * 2.5 * b + s * 0.3,
            lambda a, b, s: b * 2.5 * a + s * 0.3,
            lambda a, b, s: 2.5 * a * b + s * 0.3,
            lambda a, b, s: b * a * 2.5 + s * 0.3,
            lambda a, b, s: 2.5 * b * a + s * 0.3,
            lambda a, b, s: b * a * 2.5 + s * 0.3,
            lambda a, b, s: s * 0.3 + a * 2.5 * b,
            lambda a, b, s: s * 0.3 + b * 2.5 * a,
            lambda a, b, s: s * 0.3 + 2.5 * a * b,
            lambda a, b, s: s * 0.3 + b * a * 2.5,
            lambda a, b, s: s * 0.3 + 2.5 * b * a,
            lambda a, b, s: s * 0.3 + b * a * 2.5,
        ]
        def make_func(step_func):
            def step(loop, free):
                (s, ), (a, b) = loop, free
                out = step_func(a, b, s)
                return (out, out)
            return step
        case_id = 0
        for is_train in [True, False]:
            for step_func in step_funcs:
                case_id += 1
                _verify_while_loop(
                    func=make_func(step_func),
                    is_train=is_train,
                    is_for=False,
                    **params
                )

    def case_2(**params):
        # This is a testcase that involves non-differentiable operators
        # There is 1 output
        # There is 2 states: i, s
        step_funcs = [
            lambda in_, s, f_1: (in_ * 2) * s * f_1,
            lambda in_, s, f_1: (in_ * 2) * f_1 * s,
            lambda in_, s, f_1: s * (in_ * 2) * f_1,
            lambda in_, s, f_1: s * f_1 * (in_ * 2),
            lambda in_, s, f_1: f_1 * (in_ * 2) * s,
            lambda in_, s, f_1: f_1 * s * (in_ * 2),
            lambda in_, s, f_1: (2 * in_) * s * f_1,
            lambda in_, s, f_1: (2 * in_) * f_1 * s,
            lambda in_, s, f_1: s * (2 * in_) * f_1,
            lambda in_, s, f_1: s * f_1 * (2 * in_),
            lambda in_, s, f_1: f_1 * (2 * in_) * s,
            lambda in_, s, f_1: f_1 * s * (2 * in_),
        ]
        def make_func(step_func):
            """This simulates:
            def compute(s, inputs, f_1, length):
                outputs = []
                for i in range(length):
                    s += inputs[i] * 2 + f_1
                    outputs.append(s)
                return outputs, s
            """
            def step(loop, free):
                (i, s), (scanned, f_1, _) = loop, free
                in_ = scanned.take(i).squeeze(axis=0)
                out = step_func(in_, s, f_1)
                return (out, (i + 1, out))
            return step
        case_id = 0
        for is_train in [True, False]:
            for step_func in step_funcs:
                case_id += 1
                _verify_while_loop(
                    func=make_func(step_func),
                    max_iterations=1000,
                    is_train=is_train,
                    is_for=True,
                    **params
                )

    def case_3(length, **params):
        # This is a testcase for multiple non-differentiable operators and different ways of slicing
        # There are 2 outputs
        # There are 3 states: i, s_0, s_1
        step_funcs = [
            lambda i_0, i_1, s_0, s_1, f_0: i_0 * (i_1 * 2) * s_0 * (s_1 * 2) * f_0,
            lambda i_0, i_1, s_0, s_1, f_0: i_0 * (i_1 * 2) * s_0 * f_0 * (s_1 * 2),
            lambda i_0, i_1, s_0, s_1, f_0: i_0 * (i_1 * 2) * (s_1 * 2) * s_0 * f_0,
            lambda i_0, i_1, s_0, s_1, f_0: i_0 * (i_1 * 2) * (s_1 * 2) * f_0 * s_0,
            lambda i_0, i_1, s_0, s_1, f_0: (i_1 * 2) * i_0 * s_0 * (s_1 * 2) * f_0,
            lambda i_0, i_1, s_0, s_1, f_0: (i_1 * 2) * i_0 * s_0 * f_0 * (s_1 * 2),
            lambda i_0, i_1, s_0, s_1, f_0: (i_1 * 2) * i_0 * (s_1 * 2) * s_0 * f_0,
            lambda i_0, i_1, s_0, s_1, f_0: (i_1 * 2) * i_0 * (s_1 * 2) * f_0 * s_0,
            lambda i_0, i_1, s_0, s_1, f_0: i_0,
            lambda i_0, i_1, s_0, s_1, f_0: i_1,
            lambda i_0, i_1, s_0, s_1, f_0: s_0,
            lambda i_0, i_1, s_0, s_1, f_0: s_1,
            lambda i_0, i_1, s_0, s_1, f_0: f_0,
        ]
        def make_func(step_func):
            """This simulates:
            def compute(input_0, input_1, s_0, s_1, f_0, length):
                output_0 = []
                output_1 = []
                for i in range(length):
                    i_0 = input_0[i]
                    i_1 = input_1[length - 1 - i]
                    out = i_0 + (i_1 * 2) + s_0 + (s_1 * 2) + f_0
                    s_0 = (s_0 + out) * 1.05
                    s_1 = (s_1 - out * 0.5) * 0.95
                    output_0.append(out)
                    output_1.append(out * 1.5)
                return outputs, s_0, s_1
            """
            def step(loop, free):
                (i, s_0, s_1), (sc_0, sc_1, f_0, _) = loop, free
                i_0 = sc_0.take(i).squeeze(axis=0)
                i_1 = sc_1.take(length - 1 - i).squeeze(axis=0)
                out = step_func(i_0, i_1, s_0, s_1, f_0)
                return ([out, out * 1.5], [i + 1, (s_0 + out) * 1.05, (s_1 - out * 0.5) * 0.95])
            return step
        case_id = 0
        for is_train in [True, False]:
            for step_func in step_funcs:
                case_id += 1
                _verify_while_loop(
                    func=make_func(step_func),
                    max_iterations=1000,
                    is_train=is_train,
                    is_for=True,
                    **params
                )

    def case_4(length, single_shape, **params):
        # It is for the case that inputs & outputs are the same
        # There are 3 outputs
        # There are 4 states: i, s_0, s_1, s_2
        # i is used in both differentiable (take) and non-differentiable (+) occasions
        step_funcs = [
            lambda i_0, i_1, s_0, s_1, f_0: i_0 * (i_1 * 2) * s_0 * (s_1 * 2) * f_0,
            lambda i_0, i_1, s_0, s_1, f_0: i_0 * (i_1 * 2) * s_0 * f_0 * (s_1 * 2),
            lambda i_0, i_1, s_0, s_1, f_0: i_0 * (i_1 * 2) * (s_1 * 2) * s_0 * f_0,
            lambda i_0, i_1, s_0, s_1, f_0: i_0 * (i_1 * 2) * (s_1 * 2) * f_0 * s_0,
            lambda i_0, i_1, s_0, s_1, f_0: (i_1 * 2) * i_0 * s_0 * (s_1 * 2) * f_0,
            lambda i_0, i_1, s_0, s_1, f_0: (i_1 * 2) * i_0 * s_0 * f_0 * (s_1 * 2),
            lambda i_0, i_1, s_0, s_1, f_0: (i_1 * 2) * i_0 * (s_1 * 2) * s_0 * f_0,
            lambda i_0, i_1, s_0, s_1, f_0: (i_1 * 2) * i_0 * (s_1 * 2) * f_0 * s_0,
            lambda i_0, i_1, s_0, s_1, f_0: i_0,
            lambda i_0, i_1, s_0, s_1, f_0: i_1,
            lambda i_0, i_1, s_0, s_1, f_0: s_0,
            lambda i_0, i_1, s_0, s_1, f_0: s_1,
            lambda i_0, i_1, s_0, s_1, f_0: f_0,
        ]
        def make_func(step_func):
            """This simulates:
            def compute(input_0, input_1, s_0, s_1, s_2, f_0, length):
                # here s_2 remains untouched
                output_0 = []
                output_1 = []
                output_2 = []
                for i in range(length):
                    i_0 = input_0[i]
                    i_1 = input_1[length - 1 - i]
                    out = i_0 + (i_1 * 2) + s_0 + (s_1 * 2) + f_0
                    out = out * i * i_0 * i_1
                    s_0 = (s_0 + out) * 1.05
                    s_1 = (s_1 - out * 0.5) * 0.95
                    output_0.append(out)
                    output_1.append(f_0)
                    output_2.append(out * 1.5)
                return output_0, output_1, output_2, s_0, s_1, s_2
            """
            def step(loop, free):
                (i, s_0, s_1, s_2), (sc_0, sc_1, f_0, _) = loop, free
                i_0 = sc_0.take(i).squeeze(axis=0)
                i_1 = sc_1.take(length - 1 - i).squeeze(axis=0)
                out = step_func(i_0, i_1, s_0, s_1, f_0)
                out = out * i.reshape([1] * len(single_shape)).broadcast_to(single_shape)
                out = out * i_0 * i_1
                return ([out, f_0, out * 1.5], [i + 1, (s_0 + out) * 1.05, (s_1 - out * 0.5) * 0.95, s_2])
            return step
        case_id = 0
        for is_train in [True, False]:
            for step_func in step_funcs:
                case_id += 1
                _verify_while_loop(
                    func=make_func(step_func),
                    max_iterations=1000,
                    is_train=is_train,
                    is_for=True,
                    **params
                )

    def case_5(length, single_shape, **params):
        # It is for the case that inputs & outputs are the same
        # There are 0 outputs
        # There are 4 states: i, s_0, s_1, s_2
        # i is used in both differentiable (take) and non-differentiable (+) occasions
        step_funcs = [
            lambda i_0, i_1, s_0, s_1, f_0: i_0 * (i_1 * 2) * s_0 * (s_1 * 2) * f_0,
            lambda i_0, i_1, s_0, s_1, f_0: i_0 * (i_1 * 2) * s_0 * f_0 * (s_1 * 2),
            lambda i_0, i_1, s_0, s_1, f_0: i_0 * (i_1 * 2) * (s_1 * 2) * s_0 * f_0,
            lambda i_0, i_1, s_0, s_1, f_0: i_0 * (i_1 * 2) * (s_1 * 2) * f_0 * s_0,
            lambda i_0, i_1, s_0, s_1, f_0: (i_1 * 2) * i_0 * s_0 * (s_1 * 2) * f_0,
            lambda i_0, i_1, s_0, s_1, f_0: (i_1 * 2) * i_0 * s_0 * f_0 * (s_1 * 2),
            lambda i_0, i_1, s_0, s_1, f_0: (i_1 * 2) * i_0 * (s_1 * 2) * s_0 * f_0,
            lambda i_0, i_1, s_0, s_1, f_0: (i_1 * 2) * i_0 * (s_1 * 2) * f_0 * s_0,
            lambda i_0, i_1, s_0, s_1, f_0: i_0,
            lambda i_0, i_1, s_0, s_1, f_0: i_1,
            lambda i_0, i_1, s_0, s_1, f_0: s_0,
            lambda i_0, i_1, s_0, s_1, f_0: s_1,
            lambda i_0, i_1, s_0, s_1, f_0: f_0,
        ]
        def make_func(step_func):
            """This simulates:
            def compute(input_0, input_1, s_0, s_1, s_2, f_0, length):
                # here s_2 remains untouched
                output_0 = []
                output_1 = []
                output_2 = []
                for i in range(length):
                    i_0 = input_0[i]
                    i_1 = input_1[length - 1 - i]
                    out = i_0 + (i_1 * 2) + s_0 + (s_1 * 2) + f_0
                    out = out * i * i_0 * i_1
                    s_0 = (s_0 + out) * 1.05
                    s_1 = (s_1 - out * 0.5) * 0.95
                    output_0.append(out)
                    output_1.append(f_0)
                    output_2.append(out * 1.5)
                return output_0, output_1, output_2, s_0, s_1, s_2
            """
            def step(loop, free):
                (i, s_0, s_1, s_2), (sc_0, sc_1, f_0, _) = loop, free
                i_0 = sc_0.take(i).squeeze(axis=0)
                i_1 = sc_1.take(length - 1 - i).squeeze(axis=0)
                out = step_func(i_0, i_1, s_0, s_1, f_0)
                out = out * i.reshape([1] * len(single_shape)).broadcast_to(single_shape)
                out = out * i_0 * i_1
                return ([], [i + 1, (s_0 + out) * 1.05, (s_1 - out * 0.5) * 0.95, s_2])
            return step
        case_id = 0
        for is_train in [True, False]:
            for step_func in step_funcs:
                case_id += 1
                _verify_while_loop(
                    func=make_func(step_func),
                    max_iterations=1000,
                    is_train=is_train,
                    is_for=True,
                    **params
                )

    def case_6(length, single_shape, **params):
        # It is for the case that inputs & outputs are the same
        # There are 3 outputs
        # There are 4 states: i, s_0, s_1, s_2
        # i is used in both differentiable (take) and non-differentiable (+) occasions
        step_funcs = [
            lambda i_0, i_1, s_0, s_1, f_0: i_0 * (i_1 * 2) * s_0 * (s_1 * 2) * f_0,
            lambda i_0, i_1, s_0, s_1, f_0: i_0 * (i_1 * 2) * s_0 * f_0 * (s_1 * 2),
            lambda i_0, i_1, s_0, s_1, f_0: i_0 * (i_1 * 2) * (s_1 * 2) * s_0 * f_0,
            lambda i_0, i_1, s_0, s_1, f_0: i_0 * (i_1 * 2) * (s_1 * 2) * f_0 * s_0,
            lambda i_0, i_1, s_0, s_1, f_0: (i_1 * 2) * i_0 * s_0 * (s_1 * 2) * f_0,
            lambda i_0, i_1, s_0, s_1, f_0: (i_1 * 2) * i_0 * s_0 * f_0 * (s_1 * 2),
            lambda i_0, i_1, s_0, s_1, f_0: (i_1 * 2) * i_0 * (s_1 * 2) * s_0 * f_0,
            lambda i_0, i_1, s_0, s_1, f_0: (i_1 * 2) * i_0 * (s_1 * 2) * f_0 * s_0,
            lambda i_0, i_1, s_0, s_1, f_0: i_0,
            lambda i_0, i_1, s_0, s_1, f_0: i_1,
            lambda i_0, i_1, s_0, s_1, f_0: s_0,
            lambda i_0, i_1, s_0, s_1, f_0: s_1,
            lambda i_0, i_1, s_0, s_1, f_0: f_0,
        ]
        def make_func(step_func):
            """This simulates:
            def compute(input_0, input_1, s_0, s_1, s_2, f_0, length):
                # here s_2 remains untouched
                output_0 = []
                output_1 = []
                output_2 = []
                for i in range(length):
                    i_0 = input_0[i]
                    i_1 = input_1[length - 1 - i]
                    out = i_0 + (i_1 * 2) + s_0 + (s_1 * 2) + f_0
                    out = out * i * i_0 * i_1
                    s_0 = (s_0 + out) * 1.05
                    s_1 = (s_1 - out * 0.5) * 0.95
                    output_0.append(out)
                    output_1.append(f_0)
                    output_2.append(out * 1.5)
                return output_0, output_1, output_2, s_0, s_1, s_2
            """
            def step(loop, free):
                (i, s_0, s_1, s_2), (sc_0, sc_1, f_0, _) = loop, free
                F = mx.sym if isinstance(i, mx.sym.Symbol) else mx.nd
                i_0 = sc_0.take(i).squeeze(axis=0)
                i_1 = sc_1.take(length - 1 - i).squeeze(axis=0)
                out_0 = step_func(i_0, i_1, s_0, s_1, f_0)
                out_0 = out_0 * i.reshape([1] * len(single_shape)).broadcast_to(single_shape)
                out_1 = step_func(i_1, s_0, f_0, s_1, i_0)
                out_1 = out_1 * i.reshape([1] * len(single_shape)).broadcast_to(single_shape)
                return ([F.dot(out_0, s_2), f_0, F.dot(s_2, out_1) * 1.5], [i + 1, (s_0 + out_1) * 1.05, (s_1 - out_0 * 0.5) * 0.95, s_2])
            return step
        case_id = 0
        for is_train in [True, False]:
            for step_func in step_funcs:
                case_id += 1
                _verify_while_loop(
                    func=make_func(step_func),
                    max_iterations=1000,
                    is_train=is_train,
                    is_for=True,
                    **params
                )

    # Case 0: the simpest case
    case_0()
    # Case 1.1.*
    case_1(
        cond=make_true_cond(),
        loop_var_shapes=[
            (1, ),          # s
        ],
        free_var_shapes=[
            (1, ),          # a
            (1, ),          # b
        ],
        max_iterations=23,
    )
    # Case 1.2.*
    case_1(
        cond=make_true_cond(),
        loop_var_shapes=[
            (2, 3, 4),      # s
        ],
        free_var_shapes=[
            (2, 3, 4),      # a
            (2, 3, 4),      # b
        ],
        max_iterations=31,
    )
    # Case 1.3.*
    case_1(
        cond=make_false_cond(),
        loop_var_shapes=[
            (2, 3, 4),      # s
        ],
        free_var_shapes=[
            (2, 3, 4),      # a
            (2, 3, 4),      # b
        ],
        max_iterations=20,
    )
    # Case 2.1.*
    case_2(
        cond=make_for_cond(length=31),
        loop_var_shapes=[
            (1, ),          # i
            (2, ),          # s
        ],
        free_var_shapes=[
            (100, 2),       # scanned
            (2, ),          # f_1
            (3, 4, 5, 6),   # f_2, unused
        ],
    )
    # Case 2.2.*
    case_2(
        cond=make_for_cond(length=25),
        loop_var_shapes=[
            (1, ),          # i
            (2, ),          # s
        ],
        free_var_shapes=[
            (30, 2),        # scanned
            (2, ),          # f_1
            (3, 4, 5, 6),   # f_2, unused
        ],
    )
    # Case 3.*
    case_3(
        length=11,
        cond=make_for_cond(length=11),
        loop_var_shapes=[
            (1, ),          # i
            (2, ),          # s_0
            (2, ),          # s_1
        ],
        free_var_shapes=[
            (30, 2),        # sc_0
            (30, 2),        # sc_1
            (2, ),          # f_0
            (3, 4, 5, 6),   # f_1, unused
        ],
    )
    # Case 4.1.*
    case_4(
        length=4,
        cond=make_for_cond(length=4),
        single_shape=[5],
        loop_var_shapes=[
            (1, ),          # i
            (5, ),          # s_0
            (5, ),          # s_1
            (23, 6, 8),     # s_2
        ],
        free_var_shapes=[
            (30, 5),        # sc_0
            (30, 5),        # sc_1
            (5, ),          # f_0
            (3, 4, 5, 6),   # f_1, unused
        ],
    )
    # Case 4.2.*
    case_4(
        length=5,
        cond=make_for_cond(length=5),
        single_shape=[5, 12],
        loop_var_shapes=[
            (1, ),          # i
            (5, 12),        # s_0
            (5, 12),        # s_1
            (23, 6, 8),     # s_2
        ],
        free_var_shapes=[
            (30, 5, 12),    # sc_0
            (30, 5, 12),    # sc_1
            (5, 12),        # f_0
            (3, 4, 5, 6),   # f_1, unused
        ],
    )
    # Case 5.1.*
    case_5(
        length=4,
        cond=make_for_cond(length=4),
        single_shape=[5],
        loop_var_shapes=[
            (1, ),          # i
            (5, ),          # s_0
            (5, ),          # s_1
            (23, 6, 8),     # s_2
        ],
        free_var_shapes=[
            (30, 5),        # sc_0
            (30, 5),        # sc_1
            (5, ),          # f_0
            (3, 4, 5, 6),   # f_1, unused
        ],
    )
    # Case 5.2.*
    case_5(
        length=5,
        cond=make_for_cond(length=5),
        single_shape=[3, 4, 2],
        loop_var_shapes=[
            (1, ),          # i
            (3, 4, 2),      # s_0
            (3, 4, 2),      # s_1
            (23, 6, 8),     # s_2
        ],
        free_var_shapes=[
            (30, 3, 4, 2),  # sc_0
            (30, 3, 4, 2),  # sc_1
            (3, 4, 2),      # f_0
            (3, 4, 5, 6),   # f_1, unused
        ],
    )
    # Case 6.*
    case_6(
        length=5,
        cond=make_for_cond(length=5),
        single_shape=[5, 3],
        loop_var_shapes=[
            (1, ),          # i
            (5, 3),         # s_0
            (5, 3),         # s_1
            (3, 5),         # s_2
        ],
        free_var_shapes=[
            (30, 5, 3),     # sc_0
            (30, 5, 3),     # sc_1
            (5, 3),         # f_0
            (3, 4, 5, 6),   # f_1, unused
        ],
    )


def test_while_loop_nested():

    def _to_np_list(arrays):
        return [x.asnumpy() if x is not None else x for x in arrays]

    def _array(shape):
        return mx.nd.random.uniform(-1.0, 1.0, shape=shape)

    def inner_cond(i, j, x_sum, sc):
        return j < 2

    def inner_body(i, j, x_sum, sc):
        x_ij = sc.take(j).squeeze(axis=0)
        return (x_ij, x_ij), (i, j + 1, x_sum, sc)

    def outer_cond(i, j, x_sum, sc):
        return i < 2

    def outer_body(i, j, x_sum, sc):
        F = mx.sym if isinstance(i, mx.sym.Symbol) else mx.nd
        (x_ij, x_ji), (i_p, j_p, x_sum_p, sc_p) = F.contrib.while_loop(
            cond=inner_cond,
            func=inner_body,
            loop_vars=(i, j, x_sum, sc),
            max_iterations=2,
        )
        return (x_ij, x_ji), (i_p + 1, j_p - 2, x_sum_p, sc_p)

    def make_loop(i, j, x_sum, sc):
        F = mx.sym if isinstance(i, mx.sym.Symbol) else mx.nd
        (x_ij, x_ji), (new_i, new_j, x_sum_p, sc_p) = F.contrib.while_loop(
            cond=outer_cond,
            func=outer_body,
            loop_vars=(i, j, x_sum, sc),
            max_iterations=2,
        )
        return new_i, new_j, x_sum_p, sc_p, x_ij, x_ji

    args = {
        "i": mx.nd.array([0]),
        "j": mx.nd.array([0]),
        "x_sum": _array([5, 3]),
        "sc": _array([10, 10, 5, 3]),
    }
    args_grad = {
        "x_sum": _array([5, 3]),
        "sc": _array([10, 10, 5, 3]),
    }
    out_grad = [
        _array([1]),
        _array([1]),
        _array([5, 3]),
        _array([10, 10, 5, 3]),
        _array([2, 2, 10, 5, 3]),
        _array([2, 2, 10, 5, 3]),
    ]
    def _get_imp_result(is_train, args, args_grad, out_grad):
        args = {k: v.copy() for k, v in args.items()}
        args_grad = {k: v.copy() for k, v in args_grad.items()}
        i, j, x_sum, sc = [args[x].copy() for x in ["i", "j", "x_sum", "sc"]]
        if is_train:
            x_sum.attach_grad()
            sc.attach_grad()
        with mx.autograd.record(train_mode=is_train):
            results = make_loop(i, j, x_sum, sc)
            cat_res = mx.nd.concat(*[x.reshape(-1) for x in results], dim=0)
        if not is_train:
            return _to_np_list(results), []
        cat_grad = mx.nd.concat(*[x.reshape(-1) for x in out_grad], dim=0)
        assert cat_grad.shape == cat_res.shape
        cat_res.backward(out_grad=cat_grad)
        grads = [x_sum.grad, sc.grad]
        return _to_np_list(results), _to_np_list(grads)

    def _get_sym_result(is_train, args, args_grad, out_grad):
        args = {k: v.copy() for k, v in args.items()}
        args_grad = {k: v.copy() for k, v in args_grad.items()}
        i, j, x_sum, sc = [
            mx.sym.var("i"),
            mx.sym.var("j"),
            mx.sym.var("x_sum"),
            mx.sym.var("sc"),
        ]
        result_sym = mx.sym.Group(make_loop(i, j, x_sum, sc))
        executor = result_sym.bind(
            ctx=default_context(),
            args=args,
            args_grad=args_grad,
        )
        results = executor.forward(is_train=is_train)
        if not is_train:
            return _to_np_list(results), []
        executor.backward(out_grads=out_grad)
        grads = [executor.grad_dict["x_sum"], executor.grad_dict["sc"]]
        return _to_np_list(results), _to_np_list(grads)

    for is_train in [True, False]:
        imp_out, imp_grad = _get_imp_result(is_train=is_train, args=args, args_grad=args_grad, out_grad=out_grad)
        sym_out, sym_grad = _get_sym_result(is_train=is_train, args=args, args_grad=args_grad, out_grad=out_grad)
        assert len(imp_out) == len(sym_out)
        assert len(imp_grad) == len(sym_grad)
        for x, y in zip(imp_out, sym_out):
            assert_almost_equal(x, y)
        for x, y in zip(imp_grad, sym_grad):
            assert_almost_equal(x, y, rtol=1e-5, atol=1e-5)


def test_while_loop_rnn():
    def _array(shape):
        return mx.nd.random.uniform(-1.0, 1.0, shape=shape)

    cell_types = [mx.rnn.LSTMCell]
    num_params = [2]

    batch_size = 2
    hidden_dim = 3
    input_dim = 4
    seq_len = 3

    for cell, n_param in zip(cell_types, num_params):
        # using while_loop
        params = mx.rnn.RNNParams()
        data = mx.sym.var("data")
        iter_i = mx.sym.var("i")
        def _cond(*states):
            i = states[0]
            return i < seq_len
        def _func(*states):
            i = states[0]
            states = states[1:]
            in_ = data.take(i).squeeze(axis=0)
            rnn = cell(hidden_dim, prefix='', params=params)
            next_hidden, next_states = rnn(in_, states)
            return [next_hidden], [i + 1] + list(next_states)
        states = [mx.sym.var("s_" + str(i)) for i in range(n_param)]
        result = mx.sym.contrib.while_loop(
                    cond=_cond,
                    func=_func,
                    loop_vars=[iter_i] + states,
                    max_iterations=seq_len
                )
        result = mx.sym.Group(result[0] + result[1][1: ])
        arg_shapes, _, _ = result.infer_shape(
            data=(seq_len, batch_size, input_dim),
            s_0=(batch_size, hidden_dim),
        )
        rnn_inputs = result.list_inputs()
        args = {name: _array(arg_shapes[i]) for i, name in enumerate(rnn_inputs)}
        args_grad = {name: _array(arg_shapes[i]) for i, name in enumerate(rnn_inputs)}
        e_1 = result.bind(ctx=default_context(),
            args={name: array.copy() for name, array in args.items()},
            args_grad={name: array.copy() for name, array in args_grad.items() if name != "i"},
        )
        # using unrolled rnn
        rnn = cell(hidden_dim, prefix='')
        unroll_outs = []
        for inputs in mx.sym.split(data, num_outputs=seq_len, axis=0, squeeze_axis=True):
            h, states = rnn(inputs, states)
            unroll_outs.append(mx.sym.expand_dims(h, axis=0))
        unroll_outs = _as_list(mx.sym.concat(*unroll_outs, dim=0))
        unroll_outs.extend(states)
        result = mx.sym.Group(unroll_outs)
        e_2 = result.bind(ctx=default_context(),
            args={name: array.copy() for name, array in args.items() if name != "i"},
            args_grad={name: array.copy() for name, array in args_grad.items() if name != "i"},
        )
        for case_id in range(5):
            out_grads = [_array(arr.shape) for arr in e_1.outputs]
            args = {name: array.copy() for name, array in args.items()}
            e_1.forward(is_train=True, **args)
            e_1.backward(out_grads)
            args = {name: array.copy() for name, array in args.items() if name != "i"}
            e_2.forward(is_train=True, **args)
            e_2.backward(out_grads)
            assert len(e_1.outputs) == len(e_2.outputs)
            for x, y in zip(e_1.outputs, e_2.outputs):
                x = x.asnumpy()
                y = y.asnumpy()
                assert_almost_equal(x, y, rtol=1e-4, atol=1e-4)
            grad_keys = list(e_2.grad_dict.keys())
            e_1_grad = [e_1.grad_dict[x] for x in grad_keys]
            e_2_grad = [e_2.grad_dict[x] for x in grad_keys]
            for x, y in zip(e_1_grad, e_2_grad):
                x = x.asnumpy()
                y = y.asnumpy()
                assert_almost_equal(x, y, rtol=1e-4, atol=1e-4)


if __name__ == '__main__':
    # import nose
    # nose.runmodule()
    test_while_loop_simple_forward()
    test_while_loop_for_foreach()
    test_while_loop_nested()
    test_while_loop_rnn()
