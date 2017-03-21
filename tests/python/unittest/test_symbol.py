import copy
import os
import mxnet as mx
import numpy as np
from common import models
import pickle as pkl

def test_symbol_basic():
    mlist = []
    mlist.append(models.mlp2())
    for m in mlist:
        m.list_arguments()
        m.list_outputs()

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
    #print(net2.debug_str())

    composed = net2(fc3_data=net1, name='composed')
    #print(composed.debug_str())
    multi_out = mx.symbol.Group([composed, net1])
    assert len(multi_out.list_outputs()) == 2


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
    assert net1.get_children().get_children().list_outputs() == ['data', 'fc1_weight', 'fc1_bias']
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

def check_symbol_consistency(sym1, sym2, ctx):
    assert sym1.list_arguments() == sym2.list_arguments()
    assert sym1.list_auxiliary_states() == sym2.list_auxiliary_states()
    assert sym1.list_outputs() == sym2.list_outputs()

    mx.test_utils.check_consistency([sym1, sym2], ctx_list=[ctx, ctx])

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


if __name__ == '__main__':
    test_symbol_children()
    test_load_000800()
    test_symbol_infer_shape_var()
    test_symbol_infer_shape()
    test_symbol_infer_type()
    test_symbol_internal()
    test_symbol_basic()
    test_symbol_compose()
    test_symbol_saveload()
    test_symbol_pickle()
