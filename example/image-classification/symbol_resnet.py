'''
Reproducing https://github.com/gcr/torch-residual-networks
For image size of 32x32

References:

Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun. "Deep Residual Learning for Image Recognition"
'''
import find_mxnet
assert find_mxnet
import mxnet as mx


def get_conv(
    name,
    data,
    num_filter,
    kernel,
    stride,
    pad,
    with_relu,
    bn_momentum
):
    conv = mx.symbol.Convolution(
        name=name,
        data=data,
        num_filter=num_filter,
        kernel=kernel,
        stride=stride,
        pad=pad,
        no_bias=True
    )
    bn = mx.symbol.BatchNorm(
        name=name + '_bn',
        data=conv,
        fix_gamma=False,
        momentum=bn_momentum,
        # Same with https://github.com/soumith/cudnn.torch/blob/master/BatchNormalization.lua
        # cuDNN v5 don't allow a small eps of 1e-5
        eps=2e-5
    )
    return (
        # It's better to remove ReLU here
        # https://github.com/gcr/torch-residual-networks
        mx.symbol.Activation(name=name + '_relu', data=bn, act_type='relu')
        if with_relu else bn
    )


def make_block(
    name,
    data,
    num_filter,
    dim_match,
    bn_momentum
):
    conv1 = get_conv(
        name=name + '_conv1',
        data=data,
        num_filter=num_filter,
        kernel=(3, 3),
        stride=(1, 1) if dim_match else (2, 2),
        pad=(1, 1),
        with_relu=True,
        bn_momentum=bn_momentum
    )
    conv2 = get_conv(
        name=name + '_conv2',
        data=conv1,
        num_filter=num_filter,
        kernel=(3, 3),
        stride=(1, 1),
        pad=(1, 1),
        with_relu=False,
        bn_momentum=bn_momentum
    )
    if dim_match:
        shortcut = data
    else:
        # Like http://ethereon.github.io/netscope/#/gist/db945b393d40bfa26006
        # Test accuracy 0.922+ on CIFAR10 with 56 layers
        # shortcut = get_conv(
            # name=name + '_proj',
            # data=data,
            # num_filter=num_filter,
            # kernel=(1, 1),
            # stride=(2, 2),
            # pad=(0, 0),
            # with_relu=False,
            # bn_momentum=bn_momentum
        # )

        # Type A shortcut
        # Note we use kernel (2, 2) rather than (1, 1) and a custom initializer
        # in train_cifar10_resnet.py
        # Test accuracy 0.918 on CIFAR10 with 56 layers and kernel (1, 1)
        # TODO(Answeror): Don't know why (1, 1) got much lower accuracy
        shortcut = mx.symbol.Convolution(
            name=name + '_proj',
            data=data,
            num_filter=num_filter,
            kernel=(2, 2),
            stride=(2, 2),
            pad=(0, 0),
            no_bias=True
        )

        # Same with above, but ugly
        # Mxnet don't have nn.Padding as that in
        # https://github.com/gcr/torch-residual-networks/blob/master/residual-layers.lua
        # shortcut = mx.symbol.Pooling(
            # data=data,
            # name=name + '_pool',
            # kernel=(2, 2),
            # stride=(2, 2),
            # pool_type='avg'
        # )
        # shortcut = mx.symbol.Concat(
            # shortcut,
            # mx.symbol.minimum(shortcut + 1, 0),
            # num_args=2
        # )
    fused = shortcut + conv2
    return mx.symbol.Activation(
        name=name + '_relu',
        data=fused,
        act_type='relu'
    )


def get_body(
    data,
    num_level,
    num_block,
    num_filter,
    bn_momentum
):
    for level in range(num_level):
        for block in range(num_block):
            data = make_block(
                name='level%d_block%d' % (level + 1, block + 1),
                data=data,
                num_filter=num_filter * (2 ** level),
                dim_match=level == 0 or block > 0,
                bn_momentum=bn_momentum
            )
    return data


def get_symbol(
    num_class,
    num_level=3,
    num_block=9,
    num_filter=16,
    bn_momentum=0.9,
    pool_kernel=(8, 8)
):
    data = mx.symbol.Variable(name='data')
    # Simulate z-score normalization as that in
    # https://github.com/gcr/torch-residual-networks/blob/master/data/cifar-dataset.lua
    zscore = mx.symbol.BatchNorm(
        name='zscore',
        data=data,
        fix_gamma=True,
        momentum=bn_momentum
    )
    conv = get_conv(
        name='conv0',
        data=zscore,
        num_filter=num_filter,
        kernel=(3, 3),
        stride=(1, 1),
        pad=(1, 1),
        with_relu=True,
        bn_momentum=bn_momentum
    )
    body = get_body(
        conv,
        num_level,
        num_block,
        num_filter,
        bn_momentum
    )
    pool = mx.symbol.Pooling(data=body, kernel=pool_kernel, pool_type='avg')
    # The flatten layer seems superfluous
    flat = mx.symbol.Flatten(data=pool)
    fc = mx.symbol.FullyConnected(data=flat, num_hidden=num_class, name='fc')
    return mx.symbol.SoftmaxOutput(data=fc, name='softmax')
