import os
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

if __name__ == '__main__':
    test_ctx_group()
