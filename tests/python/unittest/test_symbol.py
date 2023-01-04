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
import sys
import os
import logging
import re
import json
import mxnet as mx
import numpy as np
from common import assertRaises, models, TemporaryDirectory
from mxnet.base import NotImplementedForSymbol
from mxnet.test_utils import discard_stderr, rand_shape_nd, use_np, environment
from mxnet.util import np_shape
import pickle as pkl

def test_symbol_basic():
    mlist = []
    mlist.append(models.mlp2())
    for m in mlist:
        m.list_arguments()
        m.list_outputs()

def test_symbol_bool():
    x = mx.symbol.Variable('x')
    assertRaises(NotImplementedForSymbol, bool, x)

def test_symbol_compose():
    data = mx.symbol.Variable('data')
    net1 = mx.symbol.FullyConnected(data=data, name='fc1', num_hidden=10)
    net1 = mx.symbol.FullyConnected(data=net1, name='fc2', num_hidden=100)
    net1.list_arguments() == ['data',
                              'fc1_weight', 'fc1_bias',
                              'fc2_weight', 'fc2_bias']

    net2 = mx.symbol.FullyConnected(name='fc3', num_hidden=10)
    net2 = mx.symbol.Activation(data=net2, act_type='relu')
    net2 = mx.symbol.FullyConnected(data=net2, name='fc4', num_hidden=20)

    composed = net2(fc3_data=net1, name='composed')
    multi_out = mx.symbol.Group([composed, net1])
    assert len(multi_out.list_outputs()) == 2
    assert len(multi_out) == 2


def test_symbol_copy():
    data = mx.symbol.Variable('data')
    data_2 = copy.deepcopy(data)
    data_3 = copy.copy(data)
    assert data.tojson() == data_2.tojson()
    assert data.tojson() == data_3.tojson()


def test_symbol_internal():
    data = mx.symbol.Variable('data')
    oldfc = mx.symbol.FullyConnected(data=data, name='fc1', num_hidden=10)
    net1 = mx.symbol.FullyConnected(data=oldfc, name='fc2', num_hidden=100)
    assert net1.list_arguments() == ['data', 'fc1_weight', 'fc1_bias', 'fc2_weight', 'fc2_bias']

    internal =  net1.get_internals()
    fc1 = internal['fc1_output']
    assert fc1.list_arguments() == oldfc.list_arguments()

def test_symbol_children():
    data = mx.symbol.Variable('data')
    oldfc = mx.symbol.FullyConnected(data=data, name='fc1', num_hidden=10)
    net1 = mx.symbol.FullyConnected(data=oldfc, name='fc2', num_hidden=100)

    assert net1.get_children().list_outputs() == ['fc1_output', 'fc2_weight', 'fc2_bias']
    assert len(net1.get_children()) == 3
    assert net1.get_children().get_children().list_outputs() == ['data', 'fc1_weight', 'fc1_bias']
    assert len(net1.get_children().get_children()) == 3
    assert net1.get_children()['fc2_weight'].list_arguments() == ['fc2_weight']
    assert net1.get_children()['fc2_weight'].get_children() is None

    data = mx.sym.Variable('data')
    sliced = mx.sym.SliceChannel(data, num_outputs=3, name='slice')
    concat = mx.sym.Concat(*list(sliced))

    assert concat.get_children().list_outputs() == \
        ['slice_output0', 'slice_output1', 'slice_output2']
    assert sliced.get_children().list_outputs() == ['data']

def test_symbol_pickle():
    mlist = [models.mlp2()]
    data = pkl.dumps(mlist)
    mlist2 = pkl.loads(data)
    for x, y  in zip(mlist, mlist2):
        assert x.tojson() == y.tojson()


def test_symbol_saveload():
    sym = models.mlp2()
    fname = 'tmp_sym.json'
    sym.save(fname)
    data2 = mx.symbol.load(fname)
    # save because of order
    assert sym.tojson() == data2.tojson()
    os.remove(fname)

def test_symbol_infer_shape():
    num_hidden = 128
    num_dim    = 64
    num_sample = 10

    data = mx.symbol.Variable('data')
    prev = mx.symbol.Variable('prevstate')
    x2h  = mx.symbol.FullyConnected(data=data, name='x2h', num_hidden=num_hidden)
    h2h  = mx.symbol.FullyConnected(data=prev, name='h2h', num_hidden=num_hidden)

    out  = mx.symbol.Activation(data=mx.sym.elemwise_add(x2h, h2h), name='out', act_type='relu')

    # shape inference will fail because information is not available for h2h
    ret  = out.infer_shape(data=(num_sample, num_dim))
    assert ret == (None, None, None)

    arg, out_shapes, aux_shapes = out.infer_shape_partial(data=(num_sample, num_dim))
    arg_shapes = dict(zip(out.list_arguments(), arg))
    assert arg_shapes['data'] == (num_sample, num_dim)
    assert arg_shapes['x2h_weight'] == (num_hidden, num_dim)
    assert arg_shapes['h2h_weight'] == ()

    # now we can do full shape inference
    state_shape = out_shapes[0]
    arg, out_shapes, aux_shapes = out.infer_shape(data=(num_sample, num_dim), prevstate=state_shape)
    arg_shapes = dict(zip(out.list_arguments(), arg))
    assert arg_shapes['data'] == (num_sample, num_dim)
    assert arg_shapes['x2h_weight'] == (num_hidden, num_dim)
    assert arg_shapes['h2h_weight'] == (num_hidden, num_hidden)

    # Partial shape inference with some unknown dimensions
    data_shape = (1, 0, 0, 0)
    data = mx.sym.Variable('data', shape=data_shape)
    weight = mx.sym.Variable('weight')
    cdata = mx.sym.cast(data, dtype='float16')
    cweight = mx.sym.cast(weight, dtype='float16')
    test = mx.sym.Convolution(data=cdata, weight=cweight, pad=(3, 3), num_filter=64, stride=(2, 2), no_bias=True, kernel=(7, 7))

    arg, _, _ = test.infer_shape_partial()
    arg_shapes = dict(zip(test.list_arguments(), arg))
    assert arg_shapes['data'] == data_shape
    assert arg_shapes['weight'] == (64, 0, 7, 7)


def test_symbol_infer_shape_var():
    "Test specifying shape information when constructing a variable"
    shape = (2, 3)
    a = mx.symbol.Variable('a', shape=shape)
    b = mx.symbol.Variable('b')
    c = mx.symbol.elemwise_add(a, b)
    arg_shapes, out_shapes, aux_shapes = c.infer_shape()
    assert arg_shapes[0] == shape
    assert arg_shapes[1] == shape
    assert out_shapes[0] == shape

    overwrite_shape = (5, 6)
    arg_shapes, out_shapes, aux_shapes = c.infer_shape(a=overwrite_shape)
    assert arg_shapes[0] == overwrite_shape
    assert arg_shapes[1] == overwrite_shape
    assert out_shapes[0] == overwrite_shape


def test_symbol_magic_abs():
    for dim in range(1, 7):
        with mx.name.NameManager():
            data = mx.symbol.Variable('data')
            method = data.abs(name='abs0')
            magic = abs(data)
            regular = mx.symbol.abs(data, name='abs0')
            ctx = {'ctx': mx.context.current_context(), 'data': rand_shape_nd(dim)}
            mx.test_utils.check_consistency(
                [method, magic], ctx_list=[ctx, ctx])
            mx.test_utils.check_consistency(
                [regular, magic], ctx_list=[ctx, ctx])


def test_symbol_fluent():
    has_grad = set(['flatten', 'expand_dims', 'flip', 'tile', 'transpose', 'sum', 'nansum', 'prod',
                    'nanprod', 'mean', 'max', 'min', 'reshape', 'broadcast_to', 'split',
                    'broadcast_axes', 'broadcast_like', 'pad', 'swapaxes', 'slice', 'slice_axis', 'slice_like',
                    'take', 'one_hot', 'pick', 'sort', 'topk', 'argsort', 'argmax', 'argmin',
                    'clip', 'abs', 'sign', 'sin', 'cos', 'tan', 'arcsin', 'arccos', 'arctan',
                    'degrees', 'radians', 'sinh', 'cosh', 'tanh', 'arcsinh', 'arccosh', 'arctanh',
                    'exp', 'expm1', 'log', 'log10', 'log2', 'log1p', 'sqrt', 'rsqrt',
                    'square', 'reciprocal' 'reshape_like', 'cbrt', 'rcbrt', 'relu', 'sigmoid',
                    'softmax', 'log_softmax', 'softmin', 'rint', 'ceil', 'floor', 'trunc', 'fix'])

    def check_fluent_regular(func, kwargs, shape=(5, 17, 1), equal_nan=False):
        with mx.name.NameManager():
            data = mx.symbol.Variable('data')
            regular = getattr(mx.symbol, func)(data, name=func+'0', **kwargs)
            fluent = getattr(data, func)(**kwargs)
            check_symbol_consistency(regular, fluent, {'ctx': mx.context.current_context(),
                                                       'data': shape},
                                     skip_grad=func not in has_grad,
                                     equal_nan=equal_nan)

    for func in ['flatten', 'norm', 'round', 'rint', 'fix', 'floor', 'ceil', 'trunc', 'zeros_like',
                 'ones_like', 'abs', 'sign', 'sin', 'cos', 'degrees', 'radians', 'exp', 'expm1',
                 'square', 'reciprocal', 'argmax_channel', 'shape_array', 'size_array']:
        check_fluent_regular(func, {})

    for func in ['arccosh', 'arcsin', 'arccos', 'arctan', 'tan', 'sinh', 'cosh', 'tanh',
                 'arcsinh', 'arctanh', 'log', 'log10', 'log2', 'log1p', 'sqrt', 'rsqrt',
                 'cbrt', 'rcbrt', 'relu', 'sigmoid', 'softmax', 'log_softmax', 'softmin']:
        check_fluent_regular(func, {}, equal_nan=True)

    for func in ['expand_dims', 'flip', 'sort', 'topk', 'argsort', 'argmax', 'argmin']:
        check_fluent_regular(func, {'axis': 1})

    check_fluent_regular('one_hot', {'depth': 15})
    check_fluent_regular('tile', {'reps': (1,2)})
    check_fluent_regular('repeat', {'repeats': 3})
    check_fluent_regular('transpose', {'axes': (1,0,2)})
    check_fluent_regular('split', {'axis': 2, 'num_outputs': 3}, shape=(5, 17, 6))
    check_fluent_regular('slice', {'begin': (2, 5, 1), 'end': (4, 7, 6)}, shape=(5, 17, 6))
    check_fluent_regular('slice_axis', {'axis': 1, 'begin': 5, 'end': 7})
    check_fluent_regular('slice_like', {'axes': (0, -2), 'shape_like': mx.sym.zeros((3, 3))})
    check_fluent_regular('clip', {'a_min': 0.25, 'a_max': 0.75})
    check_fluent_regular('broadcast_axes', {'axis': (2,), 'size': (5,)})
    check_fluent_regular('broadcast_like', {'rhs': mx.sym.ones((1, 5)), 'lhs_axes': (0,), 'rhs_axes': (1,)}, shape=(1,9))
    check_fluent_regular('pad', {'mode': 'constant', 'pad_width': (0,0,0,0,3,0,0,4)}, shape=(5, 17, 2, 3))
    check_fluent_regular('reshape_like', {'rhs': mx.sym.ones((30, 17))}, shape=(5, 17, 2, 3))

    for func in ['sum', 'nansum', 'prod', 'nanprod', 'mean', 'max', 'min', 'norm']:
        check_fluent_regular(func, {'axis': (1, 2)})

    check_fluent_regular('reshape', {'shape': (17, 1, 5)})
    check_fluent_regular('broadcast_to', {'shape': (5, 17, 47)})
    check_fluent_regular('squeeze', {'axis': (1, 3)}, shape=(2, 1, 3, 1, 4))
    check_fluent_regular('squeeze', {}, shape=(2, 1, 3, 1, 4))

def check_symbol_consistency(sym1, sym2, ctx, skip_grad=False, equal_nan=False):
    assert sym1.list_arguments() == sym2.list_arguments()
    assert sym1.list_auxiliary_states() == sym2.list_auxiliary_states()
    assert sym1.list_outputs() == sym2.list_outputs()

    mx.test_utils.check_consistency([sym1, sym2], ctx_list=[ctx, ctx],
                                    grad_req='null' if skip_grad else 'write',
                                    equal_nan=equal_nan)

def test_blockgrad():
    a = mx.sym.Variable('a')
    b = mx.sym.BlockGrad(2*a)
    exe = b._simple_bind(ctx=mx.cpu(), a=(10,10))


def test_zero_prop2():
    x = mx.sym.Variable('x')
    idx = mx.sym.Variable('idx')
    y = mx.sym.batch_take(x, idx)
    z = mx.sym.stop_gradient(y)
    exe = z._simple_bind(ctx=mx.cpu(), x=(10, 10), idx=(10,),
                        type_dict={'x': np.float32, 'idx': np.int32})
    exe.forward(is_train=True)
    exe.backward()
    mx.nd.waitall()


def test_simple_bind_incomplete_shape_inference_in_one_forward_pass():
    r"""This is a special case that results in shape inference
    failure after moving _simple_bind logic from frontend to backend.
    Added here for testing against the network similar to the following one.

    Network diagram:
    weight --> abs_op --> sum_op --
          \                        |--> add_op
    data   --> fc_op  --> sum_op --

    Given data's shape, if the shape inference starts from weight node,
    then the node entries of negative_op and sum_op are unknown in the
    forward pass. Therefore, there are several unknown shapes after the
    first forward pass is done. Now the backward inference pass starts with
    the assumption that there are no unknown-shape node entries in the forward
    pass, and consequently, leads to CHECK_EQ failure.
    """
    data_shape = (5, 13)
    data = mx.sym.Variable('data')
    fc = mx.sym.FullyConnected(data=data, num_hidden=1, no_bias=True, name='fc')
    modified_weight = mx.sym.abs(fc.get_internals()['fc_weight'])
    net = mx.sym.sum(modified_weight) + mx.sym.sum(fc)
    net._simple_bind(ctx=mx.cpu(), data=data_shape)


def test_simple_bind_gradient_graph_possible_with_cycle():
    """This is a special case that results in a cycle in the gradient graph
    before this bug was fixed. With the following symbol, the node entries
    passed into function AggregateGradient(std::vector<nnvm::NodeEntry>&& v)
    are the outputs of the same node. Therefore, adding a node to the
    control_deps of itself must be skipped.
    See GitHub issue:
    https://github.com/apache/mxnet/issues/8029
    for more details."""
    data = mx.symbol.Variable('data')
    res = data + data + data + data + data + data + data + data
    res._simple_bind(ctx=mx.cpu(), data=(1,))

def test_children_same_name():
    a = mx.sym.Variable('data')
    b = a + a
    for _ in b.get_children():
        pass

def test_transpose_nullop():
    for dim in range(1, 7):
        a = mx.sym.Variable('a')
        b = mx.sym.transpose(a, axes=tuple(np.random.permutation(dim)))
        c = mx.sym.zeros_like(b)

        shape = rand_shape_nd(dim)
        nd_a = mx.nd.random.normal(shape=shape)
        c_out = c.eval(ctx=mx.cpu(), a=nd_a)
        b_out = b.eval(ctx=mx.cpu(), a=nd_a)

        assert mx.test_utils.same(c_out[0].asnumpy(),
                                  np.zeros_like(b_out[0].asnumpy()))


def test_gen_atomic_symbol_multiple_outputs():
    data=mx.sym.Variable('data')
    p = mx.sym.Variable('param')
    h0 = mx.sym.Variable('h0')
    h1 = mx.sym.Variable('h1')
    s = mx.sym.RNN(data, p, h0, h1, state_size=10, num_layers=2,
                   bidirectional=True, state_outputs=True, mode='lstm')
    atomic_sym = s._gen_atomic_symbol()


def test_eliminate_common_expr():
    # helper function to test a single model
    def check_cse_on_symbol(sym, expected_savings, check_data, **kwargs):
        inputs = sym.list_inputs()
        shapes = {inp : kwargs[inp].shape for inp in inputs}
        rtol = {'float16' : 1e-2,
                'float32' : 1.5e-6,
                'float64' : 1.5e-6,
                }
        atol = {'float16' : 1e-3,
                'float32' : 1e-7,
                'float64' : 1e-7,
                }
        for dtype in ['float16', 'float32', 'float64']:
            data = {inp : kwargs[inp].astype(dtype) for inp in inputs}
            for grad_req in ['write', 'add']:
                type_dict = {inp : dtype for inp in inputs}
                with environment({'MXNET_ELIMINATE_COMMON_EXPR': '0'}):
                    orig_exec = sym._simple_bind(ctx=mx.cpu(0), grad_req=grad_req,
                                                type_dict=type_dict, **shapes)
                with environment({'MXNET_ELIMINATE_COMMON_EXPR': '1'}):
                    cse_exec = sym._simple_bind(ctx=mx.cpu(0), grad_req=grad_req,
                                               type_dict=type_dict, **shapes)
                fwd_orig = orig_exec.forward(is_train=True, **data)
                out_grads = [mx.nd.ones_like(arr) for arr in fwd_orig]
                orig_exec.backward(out_grads=out_grads)
                fwd_cse = cse_exec.forward(is_train=True, **data)
                cse_exec.backward(out_grads=out_grads)
                if check_data:
                    for orig, cse in zip(fwd_orig, fwd_cse):
                        np.testing.assert_allclose(orig.asnumpy(), cse.asnumpy(),
                                                   rtol=rtol[dtype], atol=atol[dtype])
                    for orig, cse in zip(orig_exec.grad_arrays, cse_exec.grad_arrays):
                        if orig is None and cse is None:
                            continue
                        assert orig is not None
                        assert cse is not None
                        np.testing.assert_allclose(orig.asnumpy(), cse.asnumpy(),
                                                   rtol=rtol[dtype], atol=atol[dtype])
                orig_sym_internals = orig_exec.get_optimized_symbol().get_internals()
                cse_sym_internals = cse_exec.get_optimized_symbol().get_internals()
                # test that the graph has been simplified as expected
                assert (len(cse_sym_internals) + expected_savings) == len(orig_sym_internals)

    a = mx.sym.Variable('a')
    b = mx.sym.Variable('b')
    c = mx.sym.Variable('c')
    shape = rand_shape_nd(2)
    arr1 = mx.random.uniform(shape=shape)
    arr2 = mx.random.uniform(shape=shape)
    arr3 = mx.random.uniform(shape=shape)

    check_cse_on_symbol((a+1) + (a+2), expected_savings=0, check_data=True, a=arr1, b=arr2)
    check_cse_on_symbol((a+b) + (a+b), expected_savings=1, check_data=True, a=arr1, b=arr2)
    check_cse_on_symbol(((a+b)+c) +((a+b)+c), expected_savings=2, check_data=True,
                                                                  a=arr1, b=arr2, c=arr3)
    d = a + 1

    # a*d node gets eliminated, but then a copy is inserted to isolate the outputs, so no net gain.
    check_cse_on_symbol(mx.sym.Group([a*d, a*d]), expected_savings=0, check_data=True, a=arr1)

    # a*d node gets eliminated, then the duplicated add-of-b, but then a copy is added for net of 1.
    check_cse_on_symbol(mx.sym.Group([a*d+b, a*d+b]), expected_savings=1, check_data=True,
                                                                          a=arr1, b=arr2)

    # dropout uses a resource that precludes any optimization
    check_cse_on_symbol(mx.sym.Dropout(a) +
                        mx.sym.Dropout(a), expected_savings=0, check_data=False, a=arr1)

def test_load_save_symbol():
    batch_size = 10
    num_hdidden = 128
    num_features = 784

    def get_net():
        data = mx.sym.var('data')
        weight = mx.sym.var('weight', shape=(num_hdidden, 0))
        return mx.sym.FullyConnected(data, weight, num_hidden=num_hdidden)

    for flag1 in [False, True]:
        with np_shape(flag1):
            net_json_str = get_net().tojson()
            net_data = json.loads(net_json_str)
            assert "attrs" in net_data
            if flag1:
                assert "is_np_shape" in net_data["attrs"]
            else:
                assert "is_np_shape" not in net_data["attrs"]

        with TemporaryDirectory() as work_dir:
            fname = os.path.join(work_dir, 'test_sym.json')
            with open(fname, 'w') as fp:
                json.dump(net_data, fp)

            # test loading 1.5.0 symbol file since 1.6.0
            # w/ or w/o np_shape semantics
            for flag2 in [False, True]:
                if flag1:  # Do not need to test this case since 0 indicates zero-size dim
                    continue
                with np_shape(flag2):
                    net = mx.sym.load(fname)
                    arg_shapes, out_shapes, aux_shapes = net.infer_shape(data=(batch_size, num_features))
                    assert arg_shapes[0] == (batch_size, num_features)  # data
                    assert arg_shapes[1] == (num_hdidden, num_features)  # weight
                    assert arg_shapes[2] == (num_hdidden,)  # bias
                    assert out_shapes[0] == (batch_size, num_hdidden)  # output
                    assert len(aux_shapes) == 0

def test_infershape_happens_for_all_ops_in_graph():
    v = mx.sym.Variable('V')
    s = mx.sym.transpose(v)
    x = mx.sym.Variable('x')
    s2 = x + v
    s3 = s + s2
    with discard_stderr():
        try:
            # This should throw an exception as you cannot add arrays
            # with shapes [2,3] and [3,2]
            e = s3._simple_bind(ctx=mx.cpu(), x=(2,3), grad_req='null')
        except:
            return

    assert False

def test_symbol_copy():
    a = mx.sym.Variable('a')
    b = copy.copy(a)
    b._set_attr(name='b')
    assert a.name == 'a' and b.name == 'b'

    a = mx.sym.Variable('a').as_np_ndarray()
    b = copy.copy(a)
    b._set_attr(name='b')
    assert a.name == 'a' and b.name == 'b'
