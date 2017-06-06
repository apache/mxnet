"""

Inception resnet v1, suitable for images with around 299 x 299

Reference:

Szegedy C, Ioffe S, Vanhoucke V. Inception-v4, inception-resnet and the impact of residual connections on learning[J]. arXiv preprint arXiv:1602.07261, 2016.

"""

import find_mxnet
import mxnet as mx
def Conv(data, num_filter, kernel=(1, 1), stride=(1, 1), pad=(0, 0), name=None, suffix='', withRelu=True, withBn=False):
    conv = mx.sym.Convolution(data=data, num_filter=num_filter, kernel=kernel, stride=stride, pad=pad,
                              name='%s%s_conv2d' % (name, suffix))
    if withBn:
        conv = mx.sym.BatchNorm(data=conv, name='%s%s_bn' % (name, suffix))
    if withRelu:
        conv = mx.sym.Activation(data=conv, act_type='relu', name='%s%s_relu' % (name, suffix))
    return conv

# Input Shape is 3*299*299 (th)
def InceptionResnetStem(data,
                        num_1_1, num_1_2, num_1_3,
                        num_2_1, num_2_2, num_2_3,
                        name):
    stem_3x3 = Conv(data=data, num_filter=num_1_1, kernel=(3, 3), stride=(2, 2), name=('%s_conv' % name))
    stem_3x3 = Conv(data=stem_3x3, num_filter=num_1_2, kernel=(3, 3), name=('%s_stem' % name), suffix='_conv_1')
    stem_3x3 = Conv(data=stem_3x3, num_filter=num_1_3, kernel=(3, 3), name=('%s_stem' % name), suffix='_conv_2')

    pool1 = mx.sym.Pooling(data=stem_3x3, kernel=(3, 3), stride=(2, 2), pool_type='max', name=('%s_%s_pool1' % ('max', name)))

    stem_1_3x3 = Conv(data=pool1, num_filter=num_2_1, name=('%s_stem_1' % name), suffix='_conv_1')
    stem_1_3x3 = Conv(data=stem_1_3x3, num_filter=num_2_2, kernel=(3, 3), name=('%s_stem_1' % name), suffix='_conv_2')
    stem_1_3x3 = Conv(data=stem_1_3x3, num_filter=num_2_3, kernel=(3, 3), pad=(1, 1), stride=(2, 2), name=('%s_stem_1' % name), suffix='_conv_3', withRelu=False)
    bn1 = mx.sym.BatchNorm(data=stem_1_3x3, name=('%s_bn1' % name))
    act1 = mx.sym.Activation(data=bn1, act_type='relu', name=('%s_relu1' % name))

    return act1


def InceptionResnetA(data,
                     num_1_1,
                     num_2_1, num_2_2,
                     num_3_1, num_3_2, num_3_3,
                     proj,
                     name,
                     scaleResidual=True):
    init = data

    a1 = Conv(data=data, num_filter=num_1_1, name=('%s_a_1' % name), suffix='_conv')

    a2 = Conv(data=data, num_filter=num_2_1, name=('%s_a_2' % name), suffix='_conv_1')
    a2 = Conv(data=a2, num_filter=num_2_2, kernel=(3, 3), pad=(1, 1), name=('%s_a_2' % name), suffix='_conv_2')

    a3 = Conv(data=data, num_filter=num_3_1, name=('%s_a_3' % name), suffix='_conv_1')
    a3 = Conv(data=a3, num_filter=num_3_2, kernel=(3, 3), pad=(1, 1), name=('%s_a_3' % name), suffix='_conv_2')
    a3 = Conv(data=a3, num_filter=num_3_3, kernel=(3, 3), pad=(1, 1), name=('%s_a_3' % name), suffix='_conv_3')

    merge = mx.sym.Concat(*[a1, a2, a3], name=('%s_a_concat1' % name))

    conv = Conv(data=merge, num_filter=proj, name=('%s_a_liner_conv' % name), withRelu=False)
    if scaleResidual:
        conv *= 0.1

    out = init + conv
    bn = mx.sym.BatchNorm(data=out, name=('%s_a_bn1' % name))
    act = mx.sym.Activation(data=bn, act_type='relu', name=('%s_a_relu1' % name))

    return act

def InceptionResnetB(data,
                     num_1_1,
                     num_2_1, num_2_2, num_2_3,
                     proj,
                     name,
                     scaleResidual=True):

    init = data

    b1 = Conv(data=data, num_filter=num_1_1, name=('%s_b_1' % name), suffix='_conv')

    b2 = Conv(data=data, num_filter=num_2_1, name=('%s_b_2' % name), suffix='_conv_1')
    b2 = Conv(data=b2, num_filter=num_2_2, kernel=(1, 7), pad=(0, 3), name=('%s_b_2' % name), suffix='_conv_2')
    b2 = Conv(data=b2, num_filter=num_2_3, kernel=(7, 1), pad=(3, 0), name=('%s_b_2' % name), suffix='_conv_3')

    merge = mx.sym.Concat(*[b1, b2], name=('%s_b_concat1' % name))

    conv = Conv(data=merge, num_filter=proj, name=('%s_b_liner_conv' % name), withRelu=False)
    if scaleResidual:
        conv *= 0.1

    out = init + conv
    bn = mx.sym.BatchNorm(data=out, name=('%s_b_bn1' % name))
    act = mx.sym.Activation(data=bn, act_type='relu', name=('%s_b_relu1' % name))

    return act

def InceptionResnetC(data,
                     num_1_1,
                     num_2_1, num_2_2, num_2_3,
                     proj,
                     name,
                     scaleResidual=True):

    init = data

    c1 = Conv(data=data, num_filter=num_1_1, name=('%s_c_1' % name), suffix='_conv')

    c2 = Conv(data=data, num_filter=num_2_1, name=('%s_c_2' % name), suffix='_conv_1')
    c2 = Conv(data=c2, num_filter=num_2_2, kernel=(1, 3), pad=(0, 1), name=('%s_c_2' % name), suffix='_conv_2')
    c2 = Conv(data=c2, num_filter=num_2_3, kernel=(3, 1), pad=(1, 0), name=('%s_c_2' % name), suffix='_conv_3')

    merge = mx.sym.Concat(*[c1, c2], name=('%s_c_concat1' % name))

    conv = Conv(data=merge, num_filter=proj, name=('%s_b_liner_conv' % name), withRelu=False)
    if scaleResidual:
        conv *= 0.1

    out = init + conv
    bn = mx.sym.BatchNorm(data=out, name=('%s_c_bn1' % name))
    act = mx.sym.Activation(data=bn, act_type='relu', name=('%s_c_relu1' % name))

    return act

def ReductionResnetA(data,
                     num_2_1,
                     num_3_1, num_3_2, num_3_3,
                     name):
    ra1 = mx.sym.Pooling(data=data, kernel=(3, 3), stride=(2, 2), pool_type='max', name=('%s_%s_pool1' % ('max', name)))

    ra2 = Conv(data=data, num_filter=num_2_1, kernel=(3, 3), stride=(2, 2), name=('%s_ra_2' % name), suffix='_conv', withRelu=False)

    ra3 = Conv(data=data, num_filter=num_3_1, name=('%s_ra_3' % name), suffix='_conv_1')
    ra3 = Conv(data=ra3, num_filter=num_3_2, kernel=(3, 3), pad=(1, 1), name=('%s_ra_3' % name), suffix='_conv_2')
    ra3 = Conv(data=ra3, num_filter=num_3_3, kernel=(3, 3), stride=(2, 2), name=('%s_ra_3' % name), suffix='_conv_3', withRelu=False)

    m = mx.sym.Concat(*[ra1, ra2, ra3], name=('%s_ra_concat1' % name))
    m = mx.sym.BatchNorm(data=m, name=('%s_ra_bn1' % name))
    m = mx.sym.Activation(data=m, act_type='relu', name=('%s_ra_relu1' % name))

    return m

def ReductionResnetB(data,
                     num_2_1, num_2_2,
                     num_3_1, num_3_2,
                     num_4_1, num_4_2, num_4_3,
                     name):
    rb1 = mx.sym.Pooling(data=data, kernel=(3, 3), stride=(2, 2), pool_type='max', name=('%s_%s_pool1' % ('max', name)))

    rb2 = Conv(data=data, num_filter=num_2_1, name=('%s_rb_2' % name), suffix='_conv_1')
    rb2 = Conv(data=rb2, num_filter=num_2_2, kernel=(3, 3), stride=(2, 2), name=('%s_rb_2' % name), suffix='_conv_2', withRelu=False)

    rb3 = Conv(data=data, num_filter=num_3_1, name=('%s_rb_3' % name), suffix='_conv_1')
    rb3 = Conv(data=rb3, num_filter=num_3_2, kernel=(3, 3), stride=(2, 2), name=('%s_rb_3' % name), suffix='_conv_2', withRelu=False)

    rb4 = Conv(data=data, num_filter=num_4_1, name=('%s_rb_4' % name), suffix='_conv_1')
    rb4 = Conv(data=rb4, num_filter=num_4_2, kernel=(3, 3), pad=(1, 1), name=('%s_rb_4' % name), suffix='_conv_2')
    rb4 = Conv(data=rb4, num_filter=num_4_3, kernel=(3, 3), stride=(2, 2), name=('%s_rb_4' % name), suffix='_conv_3', withRelu=False)

    m = mx.sym.Concat(*[rb1, rb2, rb3, rb4], name=('%s_rb_concat1' % name))
    m = mx.sym.BatchNorm(data=m, name=('%s_rb_bn1' % name))
    m = mx.sym.Activation(data=m, act_type='relu', name=('%s_rb_relu1' % name))

    return m

def circle_in3a(data,
                num_1_1,
                num_2_1, num_2_2,
                num_3_1, num_3_2, num_3_3,
                proj,
                name,
                scale,
                round):
    in3a = data
    for i in xrange(round):
        in3a = InceptionResnetA(in3a,
                                num_1_1,
                                num_2_1, num_2_2,
                                num_3_1, num_3_2, num_3_3,
                                proj,
                                name + ('_%d' % i),
                                scaleResidual=scale)
    return in3a

def circle_in2b(data,
                num_1_1,
                num_2_1, num_2_2, num_2_3,
                proj,
                name,
                scale,
                round):
    in2b = data
    for i in xrange(round):
        in2b = InceptionResnetB(in2b,
                                num_1_1,
                                num_2_1, num_2_2, num_2_3,
                                proj,
                                name + ('_%d' % i),
                                scaleResidual=scale)
    return in2b

def circle_in2c(data,
                num_1_1,
                num_2_1, num_2_2, num_2_3,
                proj,
                name,
                scale,
                round):
    in2c = data
    for i in xrange(round):
        in2c = InceptionResnetC(in2c,
                                num_1_1,
                                num_2_1, num_2_2, num_2_3,
                                proj,
                                name + ('_%d' % i),
                                scaleResidual=scale)
    return in2c

# create inception-resnet-v1
def get_symbol(num_classes=1000, scale=True):

    # input shape 3*229*229
    data = mx.symbol.Variable(name="data")

    # stage stem
    (num_1_1, num_1_2, num_1_3) = (32, 32, 64)
    (num_2_1, num_2_2, num_2_3) = (80, 192, 256)

    in_stem = InceptionResnetStem(data,
                                  num_1_1, num_1_2, num_1_3,
                                  num_2_1, num_2_2, num_2_3,
                                  'stem_stage')

    # stage 5 x Inception Resnet A
    num_1_1 = 32
    (num_2_1, num_2_2) = (32, 32)
    (num_3_1, num_3_2, num_3_3) = (32, 32, 32)
    proj = 256

    in3a = circle_in3a(in_stem,
                       num_1_1,
                       num_2_1, num_2_2,
                       num_3_1, num_3_2, num_3_3,
                       proj,
                       'in3a',
                       scale,
                       5)

    # stage Reduction Resnet A
    num_1_1 = 384
    (num_2_1, num_2_2, num_2_3) = (192, 192, 256)

    re3a = ReductionResnetA(in3a,
                            num_1_1,
                            num_2_1, num_2_2, num_2_3,
                            're3a')

    # stage 10 x Inception Resnet B
    num_1_1 = 128
    (num_2_1, num_2_2, num_2_3) = (128, 128, 128)
    proj = 896

    in2b = circle_in2b(re3a,
                       num_1_1,
                       num_2_1, num_2_2, num_2_3,
                       proj,
                       'in2b',
                       scale,
                       10)

    # stage Reduction Resnet B
    (num_1_1, num_1_2) = (256, 384)
    (num_2_1, num_2_2) = (256, 256)
    (num_3_1, num_3_2, num_3_3) = (256, 256, 256)

    re4b = ReductionResnetB(in2b,
                            num_1_1, num_1_2,
                            num_2_1, num_2_2,
                            num_3_1, num_3_2, num_3_3,
                            're4b')

    # stage 5 x Inception Resnet C
    num_1_1 = 128
    (num_2_1, num_2_2, num_2_3) = (192, 192, 192)
    proj = 1792

    in2c = circle_in2c(re4b,
                       num_1_1,
                       num_2_1, num_2_2, num_2_3,
                       proj,
                       'in2c',
                       scale,
                       5)

    # stage Average Pooling
    pool = mx.sym.Pooling(data=in2c, kernel=(8, 8), stride=(1, 1), pool_type="avg", name="global_pool")

    # stage Dropout
    dropout = mx.sym.Dropout(data=pool, p=0.2)
    # dropout =  mx.sym.Dropout(data=pool, p=0.8)
    flatten = mx.sym.Flatten(data=dropout, name="flatten")

    # output
    fc1 = mx.symbol.FullyConnected(data=flatten, num_hidden=num_classes, name='fc1')
    softmax = mx.symbol.SoftmaxOutput(data=fc1, name='softmax')
    return softmax

if __name__ == '__main__':
    net = get_symbol(1000, scale=True)
    shape = {'softmax_label': (32, 1000), 'data': (32, 3, 299, 299)}
    mx.viz.plot_network(net, title='inception-resnet-v1', format='png', shape=shape).render('inception-resnet-v1')
