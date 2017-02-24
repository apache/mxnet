import mxnet as mx

def test_print_summary():
    data = mx.sym.Variable('data')
    bias = mx.sym.Variable('fc1_bias', lr_mult=1.0)
    conv1= mx.symbol.Convolution(data = data, name='conv1', num_filter=32, kernel=(3,3), stride=(2,2))
    bn1 = mx.symbol.BatchNorm(data = conv1, name="bn1")
    act1 = mx.symbol.Activation(data = bn1, name='relu1', act_type="relu")
    mp1 = mx.symbol.Pooling(data = act1, name = 'mp1', kernel=(2,2), stride=(2,2), pool_type='max')
    fc1 = mx.sym.FullyConnected(data=mp1, bias=bias, name='fc1', num_hidden=10, lr_mult=0)
    fc2 = mx.sym.FullyConnected(data=fc1, name='fc2', num_hidden=10, wd_mult=0.5)
    sc1 = mx.symbol.SliceChannel(data=fc2, num_outputs=10, name="slice_1", squeeze_axis=0)
    mx.viz.print_summary(sc1)
    shape = {}
    shape["data"]=(1,3,28,28)
    mx.viz.print_summary(sc1, shape)

if __name__ == "__main__":
    test_print_summary()
