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

# -*- coding:utf-8 -*-
__author__ = 'zhangshuai'
modified_date = '16/7/5'
__modify__ = 'anchengwu'
modified_date = '17/2/22'

'''
Inception v4 , suittable for image with around 299 x 299

Reference:
    Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning
    Christian Szegedy, Sergey Ioffe, Vincent Vanhoucke
    arXiv.1602.07261
'''
import mxnet as mx
import numpy as np

def Conv(data, num_filter, kernel=(1, 1), stride=(1, 1), pad=(0, 0), name=None, suffix=''):
    conv = mx.sym.Convolution(data=data, num_filter=num_filter, kernel=kernel, stride=stride, pad=pad, no_bias=True, name='%s%s_conv2d' %(name, suffix))
    bn = mx.sym.BatchNorm(data=conv, name='%s%s_batchnorm' %(name, suffix), fix_gamma=True)
    act = mx.sym.Activation(data=bn, act_type='relu', name='%s%s_relu' %(name, suffix))

    return act


def Inception_stem(data, name= None):
    c = Conv(data, 32, kernel=(3, 3), stride=(2, 2), name='%s_conv1_3*3' %name)
    c = Conv(c, 32, kernel=(3, 3), name='%s_conv2_3*3' %name)
    c = Conv(c, 64, kernel=(3, 3), pad=(1, 1), name='%s_conv3_3*3' %name)

    p1 = mx.sym.Pooling(c, kernel=(3, 3), stride=(2, 2), pool_type='max', name='%s_maxpool_1' %name)
    c2 = Conv(c, 96, kernel=(3, 3), stride=(2, 2), name='%s_conv4_3*3' %name)
    concat = mx.sym.Concat(*[p1, c2], name='%s_concat_1' %name)

    c1 = Conv(concat, 64, kernel=(1, 1), pad=(0, 0), name='%s_conv5_1*1' %name)
    c1 = Conv(c1, 96, kernel=(3, 3), name='%s_conv6_3*3' %name)

    c2 = Conv(concat, 64, kernel=(1, 1), pad=(0, 0), name='%s_conv7_1*1' %name)
    c2 = Conv(c2, 64, kernel=(7, 1), pad=(3, 0), name='%s_conv8_7*1' %name)
    c2 = Conv(c2, 64, kernel=(1, 7), pad=(0, 3), name='%s_conv9_1*7' %name)
    c2 = Conv(c2, 96, kernel=(3, 3), pad=(0, 0), name='%s_conv10_3*3' %name)

    concat = mx.sym.Concat(*[c1, c2], name='%s_concat_2' %name)

    c1 = Conv(concat, 192, kernel=(3, 3), stride=(2, 2), name='%s_conv11_3*3' %name)
    p1 = mx.sym.Pooling(concat, kernel=(3, 3), stride=(2, 2), pool_type='max', name='%s_maxpool_2' %name)

    concat = mx.sym.Concat(*[c1, p1], name='%s_concat_3' %name)

    return concat


def InceptionA(input, name=None):
    p1 = mx.sym.Pooling(input, kernel=(3, 3), pad=(1, 1), pool_type='avg', name='%s_avgpool_1' %name)
    c1 = Conv(p1, 96, kernel=(1, 1), pad=(0, 0), name='%s_conv1_1*1' %name)

    c2 = Conv(input, 96, kernel=(1, 1), pad=(0, 0), name='%s_conv2_1*1' %name)

    c3 = Conv(input, 64, kernel=(1, 1), pad=(0, 0), name='%s_conv3_1*1' %name)
    c3 = Conv(c3, 96, kernel=(3, 3), pad=(1, 1), name='%s_conv4_3*3' %name)

    c4 = Conv(input, 64, kernel=(1, 1), pad=(0, 0), name='%s_conv5_1*1' % name)
    c4 = Conv(c4, 96, kernel=(3, 3), pad=(1, 1), name='%s_conv6_3*3' % name)
    c4 = Conv(c4, 96, kernel=(3, 3), pad=(1, 1), name='%s_conv7_3*3' %name)

    concat = mx.sym.Concat(*[c1, c2, c3, c4], name='%s_concat_1' %name)

    return concat


def ReductionA(input, name=None):
    p1 = mx.sym.Pooling(input, kernel=(3, 3), stride=(2, 2), pool_type='max', name='%s_maxpool_1' %name)

    c2 = Conv(input, 384, kernel=(3, 3), stride=(2, 2), name='%s_conv1_3*3' %name)

    c3 = Conv(input, 192, kernel=(1, 1), pad=(0, 0), name='%s_conv2_1*1' %name)
    c3 = Conv(c3, 224, kernel=(3, 3), pad=(1, 1), name='%s_conv3_3*3' %name)
    c3 = Conv(c3, 256, kernel=(3, 3), stride=(2, 2), pad=(0, 0), name='%s_conv4_3*3' %name)

    concat = mx.sym.Concat(*[p1, c2, c3], name='%s_concat_1' %name)

    return concat

def InceptionB(input, name=None):
    p1 = mx.sym.Pooling(input, kernel=(3, 3), pad=(1, 1), pool_type='avg', name='%s_avgpool_1' %name)
    c1 = Conv(p1, 128, kernel=(1, 1), pad=(0, 0), name='%s_conv1_1*1' %name)

    c2 = Conv(input, 384, kernel=(1, 1), pad=(0, 0), name='%s_conv2_1*1' %name)

    c3 = Conv(input, 192, kernel=(1, 1), pad=(0, 0), name='%s_conv3_1*1' %name)
    c3 = Conv(c3, 224, kernel=(1, 7), pad=(0, 3), name='%s_conv4_1*7' %name)
    #paper wrong
    c3 = Conv(c3, 256, kernel=(7, 1), pad=(3, 0), name='%s_conv5_1*7' %name)

    c4 = Conv(input, 192, kernel=(1, 1), pad=(0, 0), name='%s_conv6_1*1' %name)
    c4 = Conv(c4, 192, kernel=(1, 7), pad=(0, 3), name='%s_conv7_1*7' %name)
    c4 = Conv(c4, 224, kernel=(7, 1), pad=(3, 0), name='%s_conv8_7*1' %name)
    c4 = Conv(c4, 224, kernel=(1, 7), pad=(0, 3), name='%s_conv9_1*7' %name)
    c4 = Conv(c4, 256, kernel=(7, 1), pad=(3, 0), name='%s_conv10_7*1' %name)

    concat = mx.sym.Concat(*[c1, c2, c3, c4], name='%s_concat_1' %name)

    return concat

def ReductionB(input,name=None):
    p1 = mx.sym.Pooling(input, kernel=(3, 3), stride=(2, 2), pool_type='max', name='%s_maxpool_1' %name)

    c2 = Conv(input, 192, kernel=(1, 1), pad=(0, 0), name='%s_conv1_1*1' %name)
    c2 = Conv(c2, 192, kernel=(3, 3), stride=(2, 2), name='%s_conv2_3*3' %name)

    c3 = Conv(input, 256, kernel=(1, 1), pad=(0, 0), name='%s_conv3_1*1' %name)
    c3 = Conv(c3, 256, kernel=(1, 7), pad=(0, 3), name='%s_conv4_1*7' %name)
    c3 = Conv(c3, 320, kernel=(7, 1), pad=(3, 0), name='%s_conv5_7*1' %name)
    c3 = Conv(c3, 320, kernel=(3, 3), stride=(2, 2), name='%s_conv6_3*3' %name)

    concat = mx.sym.Concat(*[p1, c2, c3], name='%s_concat_1' %name)

    return concat


def InceptionC(input, name=None):
    p1 = mx.sym.Pooling(input, kernel=(3, 3), pad=(1, 1), pool_type='avg', name='%s_avgpool_1' %name)
    c1 = Conv(p1, 256, kernel=(1, 1), pad=(0, 0), name='%s_conv1_1*1' %name)

    c2 = Conv(input, 256, kernel=(1, 1), pad=(0, 0), name='%s_conv2_1*1' %name)

    c3 = Conv(input, 384, kernel=(1, 1), pad=(0, 0), name='%s_conv3_1*1' %name)
    c3_1 = Conv(c3, 256, kernel=(1, 3), pad=(0, 1), name='%s_conv4_3*1' %name)
    c3_2 = Conv(c3, 256, kernel=(3, 1), pad=(1, 0), name='%s_conv5_1*3' %name)

    c4 = Conv(input, 384, kernel=(1, 1), pad=(0, 0), name='%s_conv6_1*1' %name)
    c4 = Conv(c4, 448, kernel=(1, 3), pad=(0, 1), name='%s_conv7_1*3' %name)
    c4 = Conv(c4, 512, kernel=(3, 1), pad=(1, 0), name='%s_conv8_3*1' %name)
    c4_1 = Conv(c4, 256, kernel=(3, 1), pad=(1, 0), name='%s_conv9_1*3' %name)
    c4_2 = Conv(c4, 256, kernel=(1, 3), pad=(0, 1), name='%s_conv10_3*1' %name)

    concat = mx.sym.Concat(*[c1, c2, c3_1, c3_2, c4_1, c4_2], name='%s_concat' %name)

    return concat


def get_symbol(num_classes=1000, dtype='float32', **kwargs):
    data = mx.sym.Variable(name="data")
    if dtype == 'float32':
        data = mx.sym.identity(data=data, name='id')
    else:
        if dtype == 'float16':
            data = mx.sym.Cast(data=data, dtype=np.float16)
    x = Inception_stem(data, name='in_stem')

    #4 * InceptionA
    # x = InceptionA(x, name='in1A')
    # x = InceptionA(x, name='in2A')
    # x = InceptionA(x, name='in3A')
    # x = InceptionA(x, name='in4A')

    for i in range(4):
        x = InceptionA(x, name='in%dA' %(i+1))

    #Reduction A
    x = ReductionA(x, name='re1A')

    #7 * InceptionB
    # x = InceptionB(x, name='in1B')
    # x = InceptionB(x, name='in2B')
    # x = InceptionB(x, name='in3B')
    # x = InceptionB(x, name='in4B')
    # x = InceptionB(x, name='in5B')
    # x = InceptionB(x, name='in6B')
    # x = InceptionB(x, name='in7B')

    for i in range(7):
        x = InceptionB(x, name='in%dB' %(i+1))

    #ReductionB
    x = ReductionB(x, name='re1B')

    #3 * InceptionC
    # x = InceptionC(x, name='in1C')
    # x = InceptionC(x, name='in2C')
    # x = InceptionC(x, name='in3C')

    for i in range(3):
        x = InceptionC(x, name='in%dC' %(i+1))

    #Average Pooling
    x = mx.sym.Pooling(x, kernel=(8, 8), pad=(1, 1), pool_type='avg', name='global_avgpool')

    #Dropout
    x = mx.sym.Dropout(x, p=0.2)

    flatten = mx.sym.Flatten(x, name='flatten')
    fc1 = mx.sym.FullyConnected(flatten, num_hidden=num_classes, name='fc1')
    if dtype == 'float16':
        fc1 = mx.sym.Cast(data=fc1, dtype=np.float32)
    softmax = mx.sym.SoftmaxOutput(fc1, name='softmax')

    return softmax
