import mxnet as mx
def get_symbol(num_classes, **kwargs):

    data = mx.sym.Variable(name='data')
    conv1_1 = crelu(data=data, num_filter=16, kernel=(7, 7), pad=(3, 3), stride=(2, 2), name='conv1_1')    #(192*192)x3/(96*96)x32
    pool1_1 = mx.sym.Pooling(data=conv1_1, kernel=(3, 3), stride=(2, 2), pad=(1, 1), pool_type='max', name='pool1_1')  #(96*96)x32/(48*48)x32
    conv2_1 = res_crelu(data=pool1_1, middle_filter=[24, 24], num_filter=64, kernel=(3, 3), stride=(1, 1), pad=(1, 1), bsr=False, proj=True, name='conv2_1') #(48*48)x32/(48*48)x64
    conv2_2 = res_crelu(data=conv2_1, middle_filter=[24, 24], num_filter=64, kernel=(3, 3), stride=(1, 1), pad=(1, 1), bsr=True, proj=False, name='conv2_2') #(48*48)x64/(48*48)x64
    conv2_3 = res_crelu(data=conv2_2, middle_filter=[24, 24], num_filter=64, kernel=(3, 3), stride=(1, 1), pad=(1, 1), bsr=True, proj=False, name='conv2_3') #(48*48)x64/(48*48)x64
    scale3_1 = bn_scale_relu(data=conv2_3, name='bsr3_1', suffix='')
    conv3_1 = res_crelu(data=scale3_1, middle_filter=[48, 48], num_filter=128, kernel=(3, 3), stride=(2, 2), pad=(1, 1), bsr=False, proj=True, name='conv3_1') #(48*48)x64/(24*24)x128
    conv3_2 = res_crelu(data=conv3_1, middle_filter=[48, 48], num_filter=128, kernel=(3, 3), stride=(1, 1), pad=(1, 1), bsr=True, proj=False, name='conv3_2') #(24*24)x64/(24*24)x128
    conv3_3 = res_crelu(data=conv3_2, middle_filter=[48, 48], num_filter=128, kernel=(3, 3), stride=(1, 1), pad=(1, 1), bsr=True, proj=False, name='conv3_3') #(24*24)x64/(24*24)x128
    conv3_4 = res_crelu(data=conv3_3, middle_filter=[48, 48], num_filter=128, kernel=(3, 3), stride=(1, 1), pad=(1, 1), bsr=True, proj=False, name='conv3_4') #(24*24)x64/(24*24)x128
    conv4_1 = inception(data=conv3_4, middle_filter=[64, [48, 128], [24, 48, 48], 128], num_filter=256, kernel=(3, 3),stride=(2, 2), proj=True, name='conv4_1', suffix='') #(24*24)x128/(12*12)x256
    conv4_2 = inception(data=conv4_1, middle_filter=[64, [64, 128], [24, 48, 48]], num_filter=256, kernel=(3, 3), stride=(1, 1), proj=False, name='conv4_2', suffix='') #(12*12)x256/(12*12)x256
    conv4_3 = inception(data=conv4_2, middle_filter=[64, [64, 128], [24, 48, 48]], num_filter=256, kernel=(3, 3), stride=(1, 1), proj=False, name='conv4_3', suffix='') #(12*12)x256/(12*12)x256
    conv4_4 = inception(data=conv4_3, middle_filter=[64, [64, 128], [24, 48, 48]], num_filter=256, kernel=(3, 3), stride=(1, 1), proj=False, name='conv4_4', suffix='') #(12*12)x256/(12*12)x256
    conv5_1 = inception(data=conv4_4, middle_filter=[64, [96, 192], [32, 64, 64], 128], num_filter=384, kernel=(3, 3), stride=(2, 2), proj=True, name='conv5_1', suffix='') #(6*6)x256/(6*6)x384
    conv5_2 = inception(data=conv5_1, middle_filter=[64, [96, 192], [32, 64, 64]], num_filter=384, kernel=(3, 3), stride=(1, 1), proj=False, name='conv5_2', suffix='') #(6*6)x384/(6*6)x384
    conv5_3 = inception(data=conv5_2, middle_filter=[64, [96, 192], [32, 64, 64]], num_filter=384, kernel=(3, 3), stride=(1, 1), proj=False, name='conv5_3', suffix='') #(6*6)x384/(6*6)x384
    conv5_4 = inception_last(data=conv5_3, middle_filter=[64, [96, 192], [32, 64, 64]], num_filter=384, kernel=(3, 3), stride=(1, 1), proj=False, name='conv5_4', suffix='') #(6*6)x384/(6*6)x384
    bsr = bn_scale_relu(data=conv5_4, name='bsr', suffix='last')
    pool5 = mx.sym.Pooling(data=bsr, kernel=(1, 1), stride=(1, 1), pad=(0, 0), pool_type='max', name='pool5')
    flatten = mx.sym.Flatten(data=pool5, name='flatten')
    fc6 = fullconnection(data=flatten, num_hidden=4096, name='fc6', suffix='')
    fc7 = fullconnection(data=fc6, num_hidden=4096, name='fc7', suffix='')
    fc8 = mx.sym.FullyConnected(data=fc7, num_hidden=num_classes, name='fc8')
    output = mx.sym.SoftmaxOutput(data=fc8, name='softmax')
    return output

def crelu(data, num_filter, kernel, stride, pad, name=None, suffix=''):
    conv1=mx.symbol.Convolution(data=data, num_filter=num_filter, kernel=kernel, stride=stride, pad=pad, name='%s%s_conv2d' %(name, suffix))
    bn=mx.symbol.BatchNorm(data=conv1, name='%s%s_batchnorm' %(name, suffix), fix_gamma=True)
    negative=mx.symbol.negative(data=bn, name='negative')
    concat=mx.symbol.concat(bn, negative)
    net=scale_and_shift(concat, num_filter)
    act=mx.symbol.Activation(data=net, act_type='relu', name='%s%s_relu' %(name, suffix))
    return act

# use linalg_gemm instead
def scale_and_shift(data, name=None, suffix=''):
    alpha = mx.symbol.Variable(name='%s%s_alpha' %(name, suffix), shape=(1), dtype='float32', init=mx.initializer.One())
    beta = mx.symbol.Variable(name='%s%s_beta' %(name, suffix), shape=(1), dtype='float32', init = mx.initializer.Zero())
    multi = mx.symbol.broadcast_mul(data, alpha)
    add = mx.symbol.broadcast_add(multi, beta)
    return add

def res_crelu(data, middle_filter, num_filter, kernel, stride, pad, bsr, proj, name=None, suffix=''):
    if bsr:
        input = bn_scale_relu(data=data, name=name, suffix='bsr')
    else:
        input = data
    if proj:
        shortcut = mx.sym.Convolution(data=data, num_filter=num_filter, kernel=(1, 1), stride=stride, name='%s%s_proj_con2d' %(name, suffix))
    else:
        shortcut = data
    conv1 = mx.sym.Convolution(data=input, num_filter=middle_filter[0], kernel=(1, 1), stride=stride, pad=(0, 0), name='%s%s_1_con2d' %(name, suffix))
    bsr = bn_scale_relu(data=conv1, name=name, suffix='group')
    conv2 = mx.sym.Convolution(data=bsr, num_filter=middle_filter[1], kernel=kernel, stride=(1, 1), pad=pad, name='%s%s_2_con2d' %(name, suffix))
    bn = mx.sym.BatchNorm(data=conv2, name='%s%s_batchnorm' %(name, suffix), fix_gamma=True)
    negative = mx.sym.negative(data=bn, name='%s%s_negative'%(name, suffix))
    concat = mx.sym.concat(bn, negative)
    scale = scale_and_shift(data=concat, name=name, suffix=suffix)
    relu = mx.sym.Activation(data=scale, act_type='relu', name='%s%s_relu' %(name, suffix))
    conv3 = mx.sym.Convolution(data=relu, num_filter=num_filter, kernel=(1, 1), stride=(1, 1), pad=(0, 0), name='%s%s_3_con2d' %(name, suffix))
    act = conv3+shortcut
    return act

def bn_scale_relu(data, name=None, suffix=''):
    bn = mx.sym.BatchNorm(data=data, name='%s%s_batchnorm' % (name, suffix), fix_gamma=True)
    scale = scale_and_shift(data=bn, name=name, suffix=suffix)
    act = mx.sym.Activation(data=scale, act_type='relu', name='%s%s_relu' % (name, suffix))
    return act

def inception(data, middle_filter, num_filter, kernel, stride, proj, name, suffix):
    if proj:
        shortcut = mx.sym.Convolution(data=data, num_filter=num_filter, kernel=(1, 1), stride=stride, name='%s_proj' %(name))
    else:
        shortcut = data
    bsr = bn_scale_relu(data=data, name=name, suffix='bsr')
    conv_a = Conv(data=bsr, num_filter=middle_filter[0], kernel=(1, 1), stride=stride, pad=(0, 0), name=name, suffix='a')

    conv_b1 = Conv(data=bsr, num_filter=middle_filter[1][0], kernel=(1, 1), stride=stride, pad=(0, 0), name=name, suffix='b1')
    conv_b2 = Conv(data=conv_b1, num_filter=middle_filter[1][1], kernel=kernel, stride=(1, 1), pad=(1, 1), name=name, suffix='b2')

    conv_c1 = Conv(data=bsr, num_filter=middle_filter[2][0], kernel=(1, 1), stride=stride, pad=(0, 0), name=name, suffix='c1')
    conv_c2 = Conv(data=conv_c1, num_filter=middle_filter[2][1], kernel=kernel, stride=(1, 1), pad=(1, 1), name=name, suffix='c2')
    conv_c3 = Conv(data=conv_c2, num_filter=middle_filter[2][2], kernel=kernel, stride=(1, 1), pad=(1, 1), name=name, suffix='c3')

    if stride[1] > 1:
        pool_d = mx.sym.Pooling(data=bsr, kernel=kernel, stride=stride, pad=(1, 1), pool_type='max', name=name+'pool')
        conv_d = Conv(data=pool_d, num_filter=middle_filter[3], kernel=(1, 1), stride=(1, 1), pad=(0, 0), name=name, suffix='proj_conv')
        conv_concat = mx.sym.concat(conv_a, conv_b2, conv_c3, conv_d)
    else:
        conv_concat = mx.sym.concat(conv_a, conv_b2, conv_c3)
    conv = mx.sym.Convolution(data=conv_concat, num_filter=num_filter, kernel=(1, 1), stride=(1, 1), pad=(0, 0), name='%s_conv' %(name))
    output = conv+shortcut
    return output

def inception_last(data, middle_filter, num_filter, kernel, stride, proj, name, suffix):
    if proj:
        shortcut = mx.sym.Convolution(data=data, num_filter=num_filter, kernel=(1, 1), stride=stride, name='%s_proj' %(name))
    else:
        shortcut = data
    bsr = bn_scale_relu(data=data, name=name, suffix='bsr')
    conv_a = Conv(data=bsr, num_filter=middle_filter[0], kernel=(1, 1), stride=stride, pad=(0, 0), name=name, suffix='a')

    conv_b1 = Conv(data=bsr, num_filter=middle_filter[1][0], kernel=(1, 1), stride=stride, pad=(0, 0), name=name, suffix='b1')
    conv_b2 = Conv(data=conv_b1, num_filter=middle_filter[1][1], kernel=kernel, stride=(1, 1), pad=(1, 1), name=name, suffix='b2')

    conv_c1 = Conv(data=bsr, num_filter=middle_filter[2][0], kernel=(1, 1), stride=stride, pad=(0, 0), name=name, suffix='c1')
    conv_c2 = Conv(data=conv_c1, num_filter=middle_filter[2][1], kernel=kernel, stride=(1, 1), pad=(1, 1), name=name, suffix='c2')
    conv_c3 = Conv(data=conv_c2, num_filter=middle_filter[2][2], kernel=kernel, stride=(1, 1), pad=(1, 1), name=name, suffix='c3')

    if stride[1] > 1:
        pool_d = mx.sym.Pooling(data=bsr, kernel=kernel, stride=stride, pad=(1, 1), pool_type='max', name=name+'pool')
        conv_d = Conv(data=pool_d, num_filter=middle_filter[3], kernel=(1, 1), stride=(1, 1), pad=(0, 0), name=name, suffix='proj_conv')
        conv_concat = mx.sym.concat(conv_a, conv_b2, conv_c3, conv_d)
    else:
        conv_concat = mx.sym.concat(conv_a, conv_b2, conv_c3)
    conv = mx.sym.Convolution(data=conv_concat, num_filter=num_filter, kernel=(1, 1), stride=(1, 1), pad=(0, 0), name='%s_conv' %(name))
    bn = mx.sym.BatchNorm(data=conv, name='%s%s_batchnorm' % (name, suffix), fix_gamma=True)
    scale = scale_and_shift(data=bn, name=name, suffix='last')
    output = scale+shortcut
    return output


def Conv(data, num_filter=1, kernel=(1, 1), stride=(1, 1), pad=(0, 0), name=None, suffix=''):
    conv = mx.sym.Convolution(data=data, num_filter=num_filter, kernel=kernel, stride=stride, pad=pad, no_bias=True, name='%s%s_conv2d' %(name, suffix))
    bn = mx.sym.BatchNorm(data=conv, name='%s%s_batchnorm' %(name, suffix), fix_gamma=True)
    scale = scale_and_shift(bn, name='%s%s_scale_and_shift' %(name, suffix))
    act = mx.sym.Activation(data=scale, act_type='relu', name='%s%s_relu' %(name, suffix))
    return act
def fullconnection(data, num_hidden, name, suffix):
    fc = mx.sym.FullyConnected(data=data, num_hidden=num_hidden, name='%s' %(name))
    bn = mx.sym.BatchNorm(data=fc, name='%s%s_batchnorm' %(name, suffix), fix_gamma=True)
    scale = scale_and_shift(data=bn, name='%s%s_scaleshift' %(name, suffix))
    dropout = mx.sym.Dropout(data=scale, name= '%s_dropout' %(name))
    relu = mx.sym.Activation(data=dropout, act_type='relu', name='%s_relu' %(name))
    return relu
