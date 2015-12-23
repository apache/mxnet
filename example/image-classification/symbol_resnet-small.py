#!/usr/bin/env python 
'''
MSRA Paper: http://arxiv.org/pdf/1512.03385v1.pdf
'''

import mxnet as mx
def ConvFactory(data, num_filter, kernel, stride=(1, 1), pad=(0, 0), act_type = 'relu',last=False):
    conv = mx.symbol.Convolution(data = data, num_filter = num_filter, kernel = kernel, stride = stride, pad = pad)
    if last:
        return conv
    else:
        bn = mx.symbol.BatchNorm(data=conv)
        act = mx.symbol.Activation(data=bn, act_type=act_type)
        return act

def ResidualFactory(data, num_filter, diff_dim=False):
    if diff_dim:
        conv1 = ConvFactory(          data=data,  num_filter=num_filter[0], kernel=(3,3), stride=(2,2), pad=(1,1), last=False)
        conv2 = ConvFactory(          data=conv1, num_filter=num_filter[1], kernel=(3,3), stride=(1,1), pad=(1,1), last=True)
        _data = mx.symbol.Convolution(data=data,  num_filter=num_filter[1], kernel=(3,3), stride=(2,2), pad=(1,1))
        data  = _data+conv2
        bn    = mx.symbol.BatchNorm(data=data)
        act   = mx.symbol.Activation(data=bn, act_type='relu')
        return act
    else:
        _data=data
        conv1 = ConvFactory(data=data,  num_filter=num_filter[0], kernel=(3,3), stride=(1,1), pad=(1,1), last=False)
        conv2 = ConvFactory(data=conv1, num_filter=num_filter[1], kernel=(3,3), stride=(1,1), pad=(1,1), last=True)
        data  = _data+conv2
        bn    = mx.symbol.BatchNorm(data=data)
        act   = mx.symbol.Activation(data=bn, act_type='relu')
        return act

def ResidualSymbol(data, n=9):
    "stage 1"
    for i in xrange(n):
        data = ResidualFactory(data, (16, 16))
    "stage 2"
    for i in xrange(n):
        if i == 0:
            data = ResidualFactory(data, (32, 32), True)
        else:
            data = ResidualFactory(data, (32, 32))
    "stage 3"
    for i in xrange(n):
        if i == 0:
            data = ResidualFactory(data, (64, 64), True)
        else:
            data = ResidualFactory(data, (64, 64))
    return data

def get_symbol(num_classes=10):
    data    = ConvFactory(data=mx.symbol.Variable(name='data'), num_filter=16, kernel=(3,3), stride=(1,1), pad=(1,1))
    res     = ResidualSymbol(data)
    pool    = mx.symbol.Pooling(data=res, kernel=(7,7), pool_type='avg')
    flatten = mx.symbol.Flatten(data=pool, name='flatten')
    fc      = mx.symbol.FullyConnected(data=flatten, num_hidden=num_classes, name='fc1')
    softmax = mx.symbol.SoftmaxOutput(data=fc, name='softmax')
    return softmax
