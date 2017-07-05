import os
import numpy as np
import mxnet as mx

def test_ctx_group():
    with mx.AttrScope(ctx_group='stage1'):
        data = mx.symbol.Variable('data')
        fc1  = mx.symbol.FullyConnected(data = data, name='fc1', num_hidden=128)
        act1 = mx.symbol.Activation(data = fc1, name='relu1', act_type="relu")

    set_stage1 = set(act1.list_arguments())
    with mx.AttrScope(ctx_group='stage2'):
        fc2  = mx.symbol.FullyConnected(data = act1, name = 'fc2', num_hidden = 64)
        act2 = mx.symbol.Activation(data = fc2, name='relu2', act_type="relu")
        fc3  = mx.symbol.FullyConnected(data = act2, name='fc3', num_hidden=10)
        fc3 = mx.symbol.BatchNorm(fc3)
        mlp  = mx.symbol.SoftmaxOutput(data = fc3, name = 'softmax')

    set_stage2 = set(mlp.list_arguments()) - set_stage1

    group2ctx = {
        'stage1' : mx.cpu(1),
        'stage2' : mx.cpu(2)
    }

    texec = mlp.simple_bind(mx.cpu(0),
                            group2ctx=group2ctx,
                            data=(1,200))

    for arr, name in zip(texec.arg_arrays, mlp.list_arguments()):
        if name in set_stage1:
            assert arr.context == group2ctx['stage1']
        else:
            assert arr.context == group2ctx['stage2']

def check_ctx_group_sparse(lhs_stype, rhs_stype):
    with mx.AttrScope(ctx_group='stage1'):
        lhs = mx.symbol.Variable('lhs', storage_type=lhs_stype)
        rhs = mx.symbol.Variable('rhs', storage_type=rhs_stype)
        plus  = mx.symbol.elemwise_add(lhs, rhs, name='plus')

    set_stage1 = set(plus.list_arguments())
    with mx.AttrScope(ctx_group='stage2'):
        softmax  = mx.symbol.SoftmaxOutput(data = plus, name = 'softmax')

    set_stage2 = set(softmax.list_arguments()) - set_stage1

    group2ctx = {
        'stage1' : mx.cpu(1),
        'stage2' : mx.cpu(2)
    }
    texec = softmax.simple_bind(mx.cpu(0), group2ctx=group2ctx, lhs=(1,200), rhs=(1,200))

    for arr, name in zip(texec.arg_arrays, softmax.list_arguments()):
        if name in set_stage1:
            assert arr.context == group2ctx['stage1']
        else:
            assert arr.context == group2ctx['stage2']

def test_ctx_group_sparse():
    check_ctx_group_sparse('default', 'default')
    check_ctx_group_sparse('default', 'row_sparse')
    check_ctx_group_sparse('row_sparse', 'row_sparse')

if __name__ == '__main__':
    test_ctx_group()
    test_ctx_group_sparse()
