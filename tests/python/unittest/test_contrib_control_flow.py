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
from numpy.testing import assert_allclose, assert_array_equal
from mxnet.test_utils import *
from mxnet.base import _as_list
from common import with_seed


@with_seed()
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
            model.hybridize()
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
            model.hybridize()
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
            model.hybridize()
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
            model.hybridize()
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
            model.hybridize()
        (outputs, ), (result_i, result_s, _) = model(
            mx.nd.array([1], dtype="int64"), # i
            mx.nd.array([0], dtype="int64"), # s
            mx.nd.array([1], dtype="int64"), # true
        )
        assert all(outputs.asnumpy() == np.arange(1, 1001).reshape(1000, 1))
        assert result_i.asscalar() == 1001
        assert result_s.asscalar() == 500500
        # Case 2.3: a corner case, in which loop body is never executed
        model = _TestBlock(
            cond=lambda i, s, false: false,
            func=lambda i, s, false: (i, (i + 1, s + i, false)),
            max_iterations=1000,
        )
        if hybridize:
            model.hybridize()
        _, (result_i, result_s, _) = model(
            mx.nd.array([1], dtype="int64"), # i
            mx.nd.array([0], dtype="int64"), # s
            mx.nd.array([0], dtype="int64"), # false
        )
        assert result_i.asscalar() == 1
        assert result_s.asscalar() == 0


def _verify_while_loop(cond, func, loop_var_shapes, free_var_shapes, is_train, max_iterations, is_for, n_steps):

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

    def _get_imperative_result(n_steps):
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
            outputs = [x[: n_steps] for x in outputs]
            out_grads = _create_arrays(x.shape for x in outputs)  \
                      + _create_arrays(x.shape for x in final_loop_vars)
            loop_result_nd = [x * 2 for x in outputs] + [x * 3 for x in final_loop_vars]
            grads = []
            if is_train:
                cat_out = mx.nd.concat(*[x.reshape(-1) for x in loop_result_nd], dim=0)
                cat_out.backward(out_grad=mx.nd.concat(*[x.reshape(-1) for x in out_grads], dim=0))
                grads = [free_vars[i].grad for i, _ in enumerate(free_var_shapes)] \
                      + [loop_vars[i].grad for i, _ in enumerate(loop_var_shapes) if i >= loop_var_start]
            return _to_numpy_list(loop_result_nd), _to_numpy_list(grads), out_grads

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
    imp_outs, imp_grads, out_grads = _get_imperative_result(n_steps)
    sym_outs, sym_grads = _get_symbolic_result(out_grads, n_steps)
    for imp_out, sym_out in zip(imp_outs, sym_outs):
        if imp_out is None or sym_out is None:
            continue
        assert_almost_equal(imp_out, sym_out, rtol=1e-4, atol=1e-4)
    for imp_grad, sym_grad in zip(imp_grads, sym_grads):
        if imp_grad is None or sym_grad is None:
            continue
        assert_almost_equal(imp_grad, sym_grad, rtol=1e-4, atol=1e-4)


@with_seed()
def test_while_loop_for_foreach():

    def make_true_cond():
        return lambda loop_vars, _: (loop_vars[0] < 1e35).prod()

    def make_false_cond():
        return lambda loop_vars, _: (loop_vars[0] > 1e35).prod()

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
            n_steps=1,
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
        # i is used in both non-differentiable (take) and differentiable (+) occasions
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
        max_iterations=5,
        n_steps=5,
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
        max_iterations=3,
        n_steps=3,
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
        n_steps=0,
    )
    # Case 2.1.*
    case_2(
        cond=make_for_cond(length=5),
        loop_var_shapes=[
            (1, ),          # i
            (2, ),          # s
        ],
        free_var_shapes=[
            (100, 2),       # scanned
            (2, ),          # f_1
            (3, 4, 5, 6),   # f_2, unused
        ],
        n_steps=5,
    )
    # Case 2.2.*
    case_2(
        cond=make_for_cond(length=3),
        loop_var_shapes=[
            (1, ),          # i
            (2, ),          # s
        ],
        free_var_shapes=[
            (30, 2),        # scanned
            (2, ),          # f_1
            (3, 4, 5, 6),   # f_2, unused
        ],
        n_steps=3,
    )
    # Case 3.*
    case_3(
        length=5,
        cond=make_for_cond(length=5),
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
        n_steps=5,
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
        n_steps=4,
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
        n_steps=5,
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
        n_steps=4,
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
        n_steps=5,
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
        n_steps=5,
    )


@with_seed()
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
            assert_almost_equal(x, y, rtol=1e-4, atol=1e-4)
        for x, y in zip(imp_grad, sym_grad):
            assert_almost_equal(x, y, rtol=1e-4, atol=1e-4)


@with_seed()
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
        args = {name: _array(arg_shapes[i]) for i, name in enumerate(rnn_inputs) if name != "i"}
        args["i"] = mx.nd.zeros([1])
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
        for case_id in range(100):
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


class TestRNNLayer(gluon.HybridBlock):
    def __init__(self, cell_type, hidden_size, prefix=None, params=None):
        super(TestRNNLayer, self).__init__(prefix=prefix, params=params)
        self.cell = cell_type(hidden_size, prefix='rnn_')

    def hybrid_forward(self, F, inputs, states):
        out, states = F.contrib.foreach(self.cell, inputs, states)
        return out

def check_contrib_rnn(cell_type, num_states):
    batch_size = 10
    hidden_size = 100
    rnn_data = mx.nd.normal(loc=0, scale=1, shape=(5, batch_size, 50))
    state_shape = (batch_size, hidden_size)
    states = [mx.nd.normal(loc=0, scale=1, shape=state_shape) for i in range(num_states)]
    layer = TestRNNLayer(cell_type, hidden_size)
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
            {'static_alloc': True},
            {'static_alloc': True, 'static_shape': True} ]
    for config in configs:
        layer = TestRNNLayer(cell_type, hidden_size)
        layer.initialize(ctx=default_context())
        layer.hybridize(**config)
        res2 = layer(rnn_data, states)
        params2 = layer.collect_params()
        for key, val in orig_params1.items():
            params2[key].set_data(copy.deepcopy(val.data()))

        trainer = gluon.Trainer(params2, 'sgd', {'learning_rate' : 0.03})
        with mx.autograd.record():
            res2 = layer(rnn_data, states)
        assert_almost_equal(res1.asnumpy(), res2.asnumpy(), rtol=0.001, atol=0.0001)
        res2.backward()
        trainer.step(batch_size)

        for key, val in params1.items():
            weight1 = val.data()
            weight2 = params2[key].data()
            assert_almost_equal(weight1.asnumpy(), weight2.asnumpy(),
                    rtol=0.001, atol=0.0001)


@with_seed()
def test_contrib_rnn():
    cell_types = [(gluon.rnn.RNNCell, 1), (gluon.rnn.LSTMCell, 2),
            (gluon.rnn.GRUCell, 1)]
    for cell_type, num_states in cell_types:
        check_contrib_rnn(cell_type, num_states)


@with_seed()
def test_foreach():
    v3 = mx.sym.var("v0")
    v4 = mx.sym.var("v1")
    v5 = mx.sym.var("v2")
    v6 = mx.sym.var("v3")
    v7 = mx.sym.var("v4")
    v8 = mx.sym.var("v5")

    def verify_foreach(step, in_syms, state_syms, free_syms,
            in_arrs, init_states, frees, out_grads, is_train=True,
            free_vars_func=None, num_iters=1):
        step_sym = lambda in_syms, state_syms : step(in_syms, state_syms, free_syms)
        res, states = mx.sym.contrib.foreach(step_sym, in_syms, state_syms)
        out = _as_list(res)
        num_outputs = len(out)
        for i in range(num_outputs):
            out[i] = out[i] * 2
        out.extend(states)
        out = mx.sym.Group(out)
        js_1 = out.tojson()
        out = mx.sym.load_json(js_1)
        js_2 = out.tojson()
        assert js_1 == js_2
        arr_grads = []
        arg_dict = {}
        arg_grad_dict = {}
        i = 0
        for arr in _as_list(in_arrs):
            arr_grad = mx.nd.empty(arr.shape)
            arr_grads.append(arr_grad)
            arg_dict['v'+str(i)] = arr
            arg_grad_dict['v'+str(i)] = arr_grad
            i = i + 1
        for arr in init_states:
            arr_grad = mx.nd.empty(arr.shape)
            arr_grads.append(arr_grad)
            arg_dict['v'+str(i)] = arr
            arg_grad_dict['v'+str(i)] = arr_grad
            i = i + 1
        for arr in frees:
            arr_grad = mx.nd.empty(arr.shape)
            arr_grads.append(arr_grad)
            arg_dict['v'+str(i)] = arr
            arg_grad_dict['v'+str(i)] = arr_grad
            i = i + 1

        if is_train:
            e = out.bind(ctx=default_context(), args=arg_dict, args_grad=arg_grad_dict)
        else:
            e = out.bind(ctx=default_context(), args=arg_dict)
        # the inputs to forward and backward are the same so forward and backward
        # should always return the same outputs.
        for i in range(num_iters):
            e.forward(is_train=is_train)
            if (is_train):
                # backward
                tmp_grads = out_grads[0][:]
                tmp_grads.extend(out_grads[1])
                e.backward(tmp_grads)

        # Below we use imperative to reimplement foreach and compute its gradients.
        res = []
        for i in range(len(_as_list(out_grads[0]))):
            res.append([])
        for arr in _as_list(in_arrs):
            arr.attach_grad()
        for arr in init_states:
            arr.attach_grad()
        for arr in frees:
            arr.attach_grad()
        with mx.autograd.record():
            frees_imp = frees if free_vars_func is None else free_vars_func(frees)
            step_imp = lambda in_arrs, state_arrs : step(in_arrs, state_arrs, frees_imp)
            states = [mx.nd.expand_dims(s, 0) for s in init_states]
            res, states = mx.nd.contrib.foreach(step_imp, in_arrs, init_states)

            res2 = _as_list(res)
            for i in range(len(res2)):
                res2[i] = res2[i] * 2
            outs = []
            outs[:] = res2[:]
            if isinstance(states, list):
                outs.extend(states)
                states = [mx.nd.expand_dims(s, 0) for s in states]
                res2.extend(states)
            else:
                outs.append(states)
                states = mx.nd.expand_dims(states, 0)
                res2.append(states)
            if is_train:
                res = mx.nd.concat(*res2, dim=0)

        tmp_grads = out_grads[0][:]
        tmp_grads1 = [mx.nd.expand_dims(grad, 0) for grad in out_grads[1]]
        tmp_grads.extend(tmp_grads1)
        if is_train:
            res.backward(mx.nd.concat(*tmp_grads, dim=0))
        for i in range(len(outs)):
            assert e.outputs[i].shape == outs[i].shape
            assert_almost_equal(e.outputs[i].asnumpy(), outs[i].asnumpy(),
                    rtol=0.001, atol=0.0001)
        if (is_train):
            all_ins = _as_list(in_arrs)[:]
            all_ins.extend(init_states)
            all_ins.extend(frees)
            size = min(len(all_ins), len(e.grad_arrays))
            for i in range(size):
                assert_almost_equal(all_ins[i].grad.asnumpy(),
                        e.grad_arrays[i].asnumpy(),
                        rtol=0.001, atol=0.0001)

    # Test cases:
    # * graph inputs are stored in different orders.
    #   This is to test if foreach finds the data arrays and weight arrays
    #   in the right location.
    # * the number of iterations: odd or even.
    # * multiple inputs and multiple outputs.
    # * inference.
    def step1(in1, states, free):
        out = in1 * 2 + states[0] + free[0]
        return (out, [out])
    frees1 = [mx.nd.arange(2), mx.nd.arange(2) + 1]
    arrs = mx.nd.arange(6).reshape(shape=(3, 2))
    states = [mx.nd.arange(2)]
    out_grads = [[mx.nd.random.uniform(-10, 10, arrs.shape)],
            [mx.nd.random.uniform(-10, 10, states[0].shape)]]
    verify_foreach(step1, v3, [v4], [v5 + v6], arrs, states, frees1, out_grads, True,
            lambda frees : [frees[0] + frees[1]])
    verify_foreach(step1, v3, [v4], [v5 + v6], arrs, states, frees1, out_grads, False,
            lambda frees : [frees[0] + frees[1]])
    verify_foreach(step1, v3, [v4], [v5 + v6], arrs, states, frees1, out_grads, True,
            lambda frees : [frees[0] + frees[1]], 5)
    verify_foreach(step1, v3, [v4], [v5 + v6], arrs, states, frees1, out_grads, False,
            lambda frees : [frees[0] + frees[1]], 5)

    # Test the even number of iterations.
    frees = [mx.nd.random.uniform(shape=(2))]
    arrs = mx.nd.random.uniform(shape=(2, 2))
    out_grads = [[mx.nd.random.uniform(-10, 10, arrs.shape)],
            [mx.nd.random.uniform(-10, 10, states[0].shape)]]
    verify_foreach(step1, v3, [v4], [v5], arrs, states, frees, out_grads)
    verify_foreach(step1, v3, [v4], [v5], arrs, states, frees, out_grads, False)
    # Test the odd number of iterations
    arrs = mx.nd.random.uniform(shape=(3, 2))
    out_grads = [[mx.nd.random.uniform(-10, 10, arrs.shape)],
            [mx.nd.random.uniform(-10, 10, states[0].shape)]]
    verify_foreach(step1, v3, [v4], [v5], arrs, states, frees, out_grads)
    verify_foreach(step1, v3, [v4], [v5], arrs, states, frees, out_grads, False)

    # Reorder the input and state in the subgraph inputs.
    def step2(in1, states, free):
        out = states[0] + in1 * 2 + free[0]
        return (out, [out])
    # Test the even number of iterations.
    arrs = mx.nd.random.uniform(shape=(2, 2))
    out_grads = [[mx.nd.random.uniform(-10, 10, arrs.shape)],
            [mx.nd.random.uniform(-10, 10, states[0].shape)]]
    verify_foreach(step2, v3, [v4], [v5], arrs, states, frees, out_grads)
    verify_foreach(step2, v3, [v4], [v5], arrs, states, frees, out_grads, False)
    # Test the odd number of iterations.
    arrs = mx.nd.random.uniform(shape=(3, 2))
    out_grads = [[mx.nd.random.uniform(-10, 10, arrs.shape)],
            [mx.nd.random.uniform(-10, 10, states[0].shape)]]
    verify_foreach(step2, v3, [v4], [v5], arrs, states, frees, out_grads)
    verify_foreach(step2, v3, [v4], [v5], arrs, states, frees, out_grads, False)

    # Test multiple inputs and outputs.
    def step3(in1, states, free):
        out = in1[0] + in1[1] * 2 + states[0] + states[1] * 2 + free[0]
        return ([out, out], [out * 2, out * 3])
    arrs = [mx.nd.random.uniform(shape=(3, 2)), mx.nd.random.uniform(shape=(3, 2))]
    states = [mx.nd.random.uniform(shape=(2)), mx.nd.random.uniform(shape=(2))]
    out_grads = [[mx.nd.random.uniform(-10, 10, arrs[0].shape), mx.nd.random.uniform(-10, 10, arrs[1].shape)],
            [mx.nd.random.uniform(-10, 10, states[0].shape), mx.nd.random.uniform(-10, 10, states[1].shape)]]
    verify_foreach(step3, [v3, v4], [v5, v6], [v7], arrs, states, frees, out_grads)
    verify_foreach(step3, [v3, v4], [v5, v6], [v7], arrs, states, frees, out_grads, False)

    # Test multiple inputs and outputs.
    # The order of subgraph inputs doesn't match the operator inputs
    def step4(in1, states, free):
        out = in1[1] * 2 + states[0] + free[0] + states[1] * 2 + in1[0]
        return ([out, out * 2], [out * 2, out * 3])
    arrs = [mx.nd.random.uniform(shape=(3, 2)), mx.nd.random.uniform(shape=(3, 2))]
    states = [mx.nd.random.uniform(shape=(2)), mx.nd.random.uniform(shape=(2))]
    out_grads = [[mx.nd.random.uniform(-10, 10, arrs[0].shape), mx.nd.random.uniform(-10, 10, arrs[1].shape)],
            [mx.nd.random.uniform(-10, 10, states[0].shape), mx.nd.random.uniform(-10, 10, states[1].shape)]]
    verify_foreach(step4, [v3, v4], [v5, v6], [v7], arrs, states, frees, out_grads)
    verify_foreach(step4, [v3, v4], [v5, v6], [v7], arrs, states, frees, out_grads, False)

    # Test multiple inputs and outputs.
    # The data inputs and states have different shapes.
    def step5(in1, states, free):
        if isinstance(in1[0], mx.nd.NDArray):
            out1 = mx.nd.broadcast_add(states[0] + free[1], in1[1] * 2)
            out2 = mx.nd.broadcast_add(in1[0], free[0] + states[1] * 2)
        else:
            out1 = mx.sym.broadcast_add(states[0] + free[1], in1[1] * 2)
            out2 = mx.sym.broadcast_add(in1[0], free[0] + states[1] * 2)
        return ([out1, out2 * 2], [states[0] * 2, states[1] * 3])
    frees = [mx.nd.random.uniform(shape=(2)), mx.nd.random.uniform(shape=(2, 2))]
    arrs = [mx.nd.random.uniform(shape=(3, 2, 2)), mx.nd.random.uniform(shape=(3, 2))]
    states = [mx.nd.random.uniform(shape=(2, 2)), mx.nd.random.uniform(shape=(2))]
    out_grads = [[mx.nd.random.uniform(-10, 10, arrs[0].shape), mx.nd.random.uniform(-10, 10, arrs[0].shape)],
            [mx.nd.random.uniform(-10, 10, states[0].shape), mx.nd.random.uniform(-10, 10, states[1].shape)]]
    verify_foreach(step5, [v3, v4], [v5, v6], [v7, v8], arrs, states, frees, out_grads, False)

    # Test multiple inputs and outputs.
    # The data inputs and states have different shapes and data types.
    def step6(in1, states, free):
        if isinstance(in1[0], mx.nd.NDArray):
            out1 = mx.nd.broadcast_add(states[0] + mx.nd.cast(free[1], 'float32'),
                    mx.nd.cast(in1[1], 'float32') * 2)
            out2 = mx.nd.broadcast_add(in1[0],
                    free[0] + mx.nd.cast(states[1], 'float32') * 2)
        else:
            out1 = mx.sym.broadcast_add(states[0] + mx.sym.cast(free[1], 'float32'),
                    mx.sym.cast(in1[1], 'float32') * 2)
            out2 = mx.sym.broadcast_add(in1[0],
                    free[0] + mx.sym.cast(states[1], 'float32') * 2)
        return ([out1, out2 * 2], [states[0] * 2, states[1] * 3])
    frees = [mx.nd.random.uniform(shape=(2)),
            mx.nd.cast(mx.nd.random.uniform(shape=(2, 2)), 'float64')]
    arrs = [mx.nd.random.uniform(shape=(3, 2, 2)),
            mx.nd.cast(mx.nd.random.uniform(shape=(3, 2)), dtype='float16')]
    states = [mx.nd.random.uniform(shape=(2, 2)),
            mx.nd.cast(mx.nd.random.uniform(shape=(2)), dtype='int32')]
    out_grads = [[mx.nd.random.uniform(-10, 10, arrs[0].shape), mx.nd.random.uniform(-10, 10, arrs[0].shape)],
            [mx.nd.random.uniform(-10, 10, states[0].shape), mx.nd.random.uniform(-10, 10, states[1].shape)]]
    verify_foreach(step6, [v3, v4], [v5, v6], [v7, v8], arrs, states, frees, out_grads, False)

    # Test multiple inputs and outputs.
    # some of the inputs are used twice.
    def step7(in1, states, free):
        out1 = states[0] + in1[0] + free[1] + in1[1] * 2 + free[0]
        out2 = in1[0] + free[0] + states[1] * 2 + in1[1]
        return ([out1, out2 * 2], [states[0] * 2, states[1] * 3])
    frees = [mx.nd.random.uniform(shape=(2)), mx.nd.random.uniform(shape=(2))]
    arrs = [mx.nd.random.uniform(shape=(3, 2)), mx.nd.random.uniform(shape=(3, 2))]
    states = [mx.nd.random.uniform(shape=(2)), mx.nd.random.uniform(shape=(2))]
    out_grads = [[mx.nd.random.uniform(-10, 10, arrs[0].shape), mx.nd.random.uniform(-10, 10, arrs[0].shape)],
            [mx.nd.random.uniform(-10, 10, states[0].shape), mx.nd.random.uniform(-10, 10, states[1].shape)]]
    verify_foreach(step7, [v3, v4], [v5, v6], [v7, v8], arrs, states, frees, out_grads, False)

    # Test the case that the output is the input.
    arrs = mx.nd.random.uniform(shape=(3, 2))
    states = [mx.nd.arange(2)]
    frees = [mx.nd.random.uniform(shape=(2))]
    out_grads = [[mx.nd.random.uniform(-10, 10, arrs.shape)],
            [mx.nd.random.uniform(-10, 10, states[0].shape)]]
    def step8(in1, states, free):
        return (in1, [states[0] * free[0]])
    verify_foreach(step8, v3, [v4], [v5], arrs, states, frees, out_grads)
    verify_foreach(step8, v3, [v4], [v5], arrs, states, frees, out_grads, False)
    def step9(in1, states, free):
        return (in1 * free[0], states)
    verify_foreach(step9, v3, [v4], [v5], arrs, states, frees, out_grads)
    verify_foreach(step9, v3, [v4], [v5], arrs, states, frees, out_grads, False)

    # Test the case that not all inputs are used.
    def step10(in1, states, free):
        return (in1, states)
    verify_foreach(step10, v3, [v4], [v5], arrs, states, frees, out_grads)
    verify_foreach(step10, v3, [v4], [v5], arrs, states, frees, out_grads, False)
    def step11(in1, states, free):
        return (in1, free)
    try:
        verify_foreach(step11, v3, [v4], [v5], arrs, states, frees, out_grads)
        verify_foreach(step11, v3, [v4], [v5], arrs, states, frees, out_grads, False)
    except AssertionError:
        print("the states have to be used")
    def step12(in1, states, free):
        return (in1, [states[0] + 1, states[0] + 2])
    states = [mx.nd.random.uniform(shape=(2)), mx.nd.random.uniform(shape=(2))]
    frees = []
    try:
        verify_foreach(step12, v3, [v4, v5], [], arrs, states, frees, out_grads)
        verify_foreach(step12, v3, [v4, v5], [], arrs, states, frees, out_grads, False)
    except AssertionError:
        print("the states have to be used")

    # test without free variables.
    def step13(in1, states, free):
        return (in1, states)
    states = [mx.nd.random.uniform(shape=(2))]
    verify_foreach(step13, v3, [v4], [], arrs, states, [], out_grads)
    verify_foreach(step13, v3, [v4], [], arrs, states, [], out_grads, False)

    # test when there isn't output data or output states.
    def step14(in1, states, free):
        return (in1 + free[0], [])
    frees = [mx.nd.random.uniform(shape=(2))]
    verify_foreach(step14, v3, [], [v4], arrs, [], frees, out_grads)
    verify_foreach(step14, v3, [], [v4], arrs, [], frees, out_grads, False)
    def step15(in1, states, free):
        return ([], [in1 * states[0] * free[0]])
    out_grads = [[], [mx.nd.random.uniform(-10, 10, states[0].shape)]]
    verify_foreach(step15, v3, [v4], [v5], arrs, states, frees, out_grads)
    verify_foreach(step15, v3, [v4], [v5], arrs, states, frees, out_grads, False)

    # Test the case of iterating on a 1D data array.
    def step16(in1, states, free):
        return ([in1[0] * states[0]], [states[0] * 2])
    arrs = [mx.nd.arange(3)]
    states = [mx.nd.random.uniform(shape=(1))]
    out_grads = [[mx.nd.random.uniform(-10, 10, (3, 1))],
            [mx.nd.random.uniform(-10, 10, (1))]]
    verify_foreach(step16, [v3], [v4], [], arrs, states, [], out_grads)
    verify_foreach(step16, [v3], [v4], [], arrs, states, [], out_grads, False)
    def step17(in1, states, free):
        return ([in1[1] * in1[0] * states[0]], [states[0] * 2])
    arrs = [mx.nd.random.uniform(shape=(3, 1)), mx.nd.arange(3)]
    states = [mx.nd.random.uniform(shape=(1))]
    out_grads = [[mx.nd.random.uniform(-10, 10, (3, 1))],
            [mx.nd.random.uniform(-10, 10, (1))]]
    verify_foreach(step17, [v3, v4], [v5], [], arrs, states, [], out_grads)
    verify_foreach(step17, [v3, v4], [v5], [], arrs, states, [], out_grads, False)


@with_seed()
def test_foreach_nested():
    # Test nested foreach.
    def step_in(in1, states):
        out = in1 * 2 + states[0]
        return (out, [out])

    def step_sym(in1, states):
        out1 = mx.sym.contrib.foreach(step_in, in1, states)
        out = mx.sym.broadcast_add(out1[0], states[0])
        return (out, [mx.sym.squeeze(mx.sym.slice(out, begin=(0, 0), end=(1, 2)))])
    def step_nd(in1, states):
        out1 = mx.nd.contrib.foreach(step_in, in1, states)
        out = mx.nd.broadcast_add(out1[0], states[0])
        return (out, [mx.nd.squeeze(mx.nd.slice(out, begin=(0, 0), end=(1, 2)))])

    data_sym = mx.sym.var("v1")
    state_sym = mx.sym.var("v2")
    out, states = mx.sym.contrib.foreach(step_sym, data_sym, [state_sym])
    assert isinstance(states, list)
    assert len(states) == 1
    out = mx.sym.broadcast_add(out, states[0])

    js_1 = out.tojson()
    out = mx.sym.load_json(js_1)
    js_2 = out.tojson()
    assert js_1 == js_2

    data = mx.nd.arange(8).reshape((2, 2, 2))
    state = mx.nd.arange(2)
    data_grad = mx.nd.empty(data.shape)
    state_grad = mx.nd.empty(state.shape)
    e = out.bind(ctx=default_context(), args={'v1':data, 'v2':state},
            args_grad={'v1':data_grad, 'v2':state_grad})
    e.forward(is_train=True)
    out_grads = []
    for out in e.outputs:
        out_grads.append(mx.nd.random.uniform(shape=out.shape))
    e.backward(out_grads)

    data.attach_grad()
    state.attach_grad()
    with mx.autograd.record():
        out, states = mx.nd.contrib.foreach(step_nd, data, [state])
        assert isinstance(states, list)
        assert len(states) == 1
        res = mx.nd.broadcast_add(out, states[0])
    assert_almost_equal(res.asnumpy(), e.outputs[0].asnumpy(), rtol=0.001, atol=0.0001)

    res.backward(out_grads[0])
    assert_almost_equal(data.grad.asnumpy(), data_grad.asnumpy())
    assert_almost_equal(state.grad.asnumpy(), state_grad.asnumpy())


def check_foreach_rnn(cell_type, num_states):
    data = mx.sym.var("data")
    params = mx.rnn.RNNParams()
    hidden_dim = 4
    input_dim = 5
    seq_len = 2
    batch_size = 2

    # This tests foreach with accumulation sum.
    def step(in1, states):
        rnn = cell_type(hidden_dim, prefix='', params=params)
        next_h, states = rnn(in1, states)
        return (next_h, states)

    def sym_group(out):
        if (isinstance(out[0], mx.sym.Symbol)):
            ret = [out[0]]
        else:
            ret = out[0]
        ret.extend(out[1])
        return mx.sym.Group(ret)

    rnn = cell_type(hidden_dim, prefix='', params=params)
    if num_states == 2:
        init_states = [mx.sym.var("h"), mx.sym.var("c")]
    else:
        init_states = [mx.sym.var("h")]
    out = mx.sym.contrib.foreach(step, data, init_states)
    out = sym_group(out)
    arg_shapes, out_shapes, aux_shapes = out.infer_shape(data=(seq_len, batch_size, input_dim),
            h=(batch_size, hidden_dim))
    rnn_inputs = out.list_inputs()

    # Inputs
    args1 = {name:mx.nd.random.uniform(shape=arg_shapes[i]) for i, name in enumerate(rnn_inputs)}
    args2 = copy.deepcopy(args1)
    # gradients for the backward of the foreach symbol
    args_grad1 = {name:mx.nd.empty(shape=arg_shapes[i]) for i, name in enumerate(rnn_inputs)}
    # gradients for the backward of the unrolled symbol.
    args_grad2 = {name:mx.nd.empty(shape=arg_shapes[i]) for i, name in enumerate(rnn_inputs)}

    # Symbol of running LSTM with foreach.
    out = mx.sym.contrib.foreach(step, data, init_states)
    out = sym_group(out)
    js_1 = out.tojson()
    out = mx.sym.load_json(js_1)
    js_2 = out.tojson()
    assert js_1 == js_2
    e1 = out.bind(ctx=default_context(), args=args1, args_grad=args_grad1)

    # Symbol of running unrolled LSTM.
    lstm = cell_type(hidden_dim, prefix='')
    unroll_outs = []
    states = init_states
    for inputs in mx.sym.split(data, num_outputs=seq_len, axis=0, squeeze_axis=True):
        h, states = lstm(inputs, states)
        unroll_outs.append(mx.sym.expand_dims(h, axis=0))
    unroll_outs = _as_list(mx.sym.concat(*unroll_outs, dim=0))
    unroll_outs.extend(states)
    out = mx.sym.Group(unroll_outs)
    js_1 = out.tojson()
    out = mx.sym.load_json(js_1)
    js_2 = out.tojson()
    assert js_1 == js_2
    e2 = out.bind(ctx=default_context(), args=args2, args_grad=args_grad2)

    for i in range(5):
        out_grads = []
        for arr in e1.outputs:
            out_grads.append(mx.nd.random.uniform(-10, 10, arr.shape))

        args = {name:mx.nd.random.uniform(shape=arg_shapes[i]) for i, name in enumerate(rnn_inputs)}

        e1.forward(is_train=True, **args)
        outputs1 = e1.outputs
        e1.backward(out_grads)

        e2.forward(is_train=True, **args)
        outputs2 = e2.outputs
        e2.backward(out_grads)

        for i in range(len(outputs2)):
            assert_almost_equal(outputs1[i].asnumpy(), outputs2[i].asnumpy(),
                    rtol=0.001, atol=0.0001)
        input_names = out.list_inputs()
        for i in range(len(e1.grad_arrays)):
            name = input_names[i]
            assert_almost_equal(args_grad1[name].asnumpy(), args_grad2[name].asnumpy(),
                    rtol=0.001, atol=0.0001)


@with_seed()
def test_foreach_rnn():
    cell_types = [(mx.rnn.LSTMCell, 2), (mx.rnn.RNNCell, 1), (mx.rnn.GRUCell, 1)]
    for cell_type, num_states in cell_types:
        check_foreach_rnn(cell_type, num_states)


if __name__ == '__main__':
    import nose
    nose.runmodule()
