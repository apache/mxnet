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
import os
import re
import mxnet as mx
import numpy as np
from common import assertRaises, models
from mxnet.base import NotImplementedForSymbol
from mxnet.test_utils import discard_stderr
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
    mlist = [models.mlp2(), models.conv()]
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

def test_symbol_infer_type():
    data = mx.symbol.Variable('data')
    f32data = mx.symbol.Cast(data=data, dtype='float32')
    fc1  = mx.symbol.FullyConnected(data = f32data, name='fc1', num_hidden=128)
    mlp  = mx.symbol.SoftmaxOutput(data = fc1, name = 'softmax')

    arg, out, aux = mlp.infer_type(data=np.float16)
    assert arg == [np.float16, np.float32, np.float32, np.float32]
    assert out == [np.float32]
    assert aux == []

    # partial infer type
    arg, out, aux = mlp.infer_type_partial()
    assert arg == [None, np.float32, np.float32, np.float32]
    assert out == [np.float32]
    assert aux == []


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

def check_symbol_consistency(sym1, sym2, ctx, skip_grad=False, equal_nan=False):
    assert sym1.list_arguments() == sym2.list_arguments()
    assert sym1.list_auxiliary_states() == sym2.list_auxiliary_states()
    assert sym1.list_outputs() == sym2.list_outputs()

    mx.test_utils.check_consistency([sym1, sym2], ctx_list=[ctx, ctx],
                                    grad_req='null' if skip_grad else 'write',
                                    equal_nan=equal_nan)

def test_load_000800():
    with mx.AttrScope(ctx_group='stage1'):
        data = mx.symbol.Variable('data', lr_mult=0.2)
        weight = mx.sym.Variable(name='fc1_weight', lr_mult=1.2)
        fc1  = mx.symbol.FullyConnected(data = data, weight=weight, name='fc1', num_hidden=128, wd_mult=0.3)
        act1 = mx.symbol.Activation(data = fc1, name='relu1', act_type="relu")

    set_stage1 = set(act1.list_arguments())
    with mx.AttrScope(ctx_group='stage2'):
        fc2  = mx.symbol.FullyConnected(data = act1, name = 'fc2', num_hidden = 64, lr_mult=0.01)
        act2 = mx.symbol.Activation(data = fc2, name='relu2', act_type="relu")
        fc3  = mx.symbol.FullyConnected(data = act2, name='fc3', num_hidden=10)
        fc3 = mx.symbol.BatchNorm(fc3, name='batchnorm0')
        sym1  = mx.symbol.SoftmaxOutput(data = fc3, name = 'softmax')

    curr_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
    sym2 = mx.sym.load(os.path.join(curr_path, 'save_000800.json'))

    attr1 = sym1.attr_dict()
    attr2 = sym2.attr_dict()
    for k, v1 in attr1.items():
        assert k in attr2, k
        v2 = attr2[k]
        for kk, vv1 in v1.items():
            if kk.startswith('__') and kk.endswith('__'):
                assert kk in v2 and v2[kk] == vv1, k + str(v1) + str(v2)

    check_symbol_consistency(sym1, sym2,
        {'ctx': mx.cpu(0), 'group2ctx': {'stage1' : mx.cpu(1), 'stage2' : mx.cpu(2)}, 'data': (1,200)})


def test_blockgrad():
    a = mx.sym.Variable('a')
    b = mx.sym.BlockGrad(2*a)
    exe = b.simple_bind(ctx=mx.cpu(), a=(10,10))


def test_zero_prop():
    data = mx.symbol.Variable('data')
    for i in range(10):
        data = data * data

    exe = data.simple_bind(ctx=mx.cpu(), data=(10, 3, 256, 256))
    big = int(re.search('Total (\d+) MB allocated', exe.debug_str()).group(1))

    exe = data.simple_bind(ctx=mx.cpu(), data=(10, 3, 256, 256), grad_req='null')
    small1 = int(re.search('Total (\d+) MB allocated', exe.debug_str()).group(1))

    data = mx.sym.stop_gradient(data)
    exe = data.simple_bind(ctx=mx.cpu(), data=(10, 3, 256, 256))
    small2 = int(re.search('Total (\d+) MB allocated', exe.debug_str()).group(1))

    assert big > small2
    assert small1 == small2

def test_zero_prop2():
    x = mx.sym.Variable('x')
    idx = mx.sym.Variable('idx')
    y = mx.sym.batch_take(x, idx)
    z = mx.sym.stop_gradient(y)
    exe = z.simple_bind(ctx=mx.cpu(), x=(10, 10), idx=(10,),
                        type_dict={'x': np.float32, 'idx': np.int32})
    exe.forward()
    exe.backward()

    # The following bind() should throw an exception. We discard the expected stderr
    # output for this operation only in order to keep the test logs clean.
    with discard_stderr():
        try:
            y.simple_bind(ctx=mx.cpu(), x=(10, 10), idx=(10,),
                          type_dict={'x': np.float32, 'idx': np.int32})
        except:
            return

    assert False


def test_simple_bind_incomplete_shape_inference_in_one_forward_pass():
    """This is a special case that results in shape inference
    failure after moving simple_bind logic from frontend to backend.
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
    net.simple_bind(ctx=mx.cpu(), data=data_shape)


def test_simple_bind_gradient_graph_possible_with_cycle():
    """This is a special case that results in a cycle in the gradient graph
    before this bug was fixed. With the following symbol, the node entries
    passed into function AggregateGradient(std::vector<nnvm::NodeEntry>&& v)
    are the outputs of the same node. Therefore, adding a node to the
    control_deps of itself must be skipped.
    See GitHub issue:
    https://github.com/apache/incubator-mxnet/issues/8029
    for more details."""
    data = mx.symbol.Variable('data')
    res = data + data + data + data + data + data + data + data
    res.simple_bind(ctx=mx.cpu(), data=(1,))

def test_children_same_name():
    a = mx.sym.Variable('data')
    b = a + a
    for c in b.get_children():
        pass

if __name__ == '__main__':
    import nose
    nose.runmodule()
