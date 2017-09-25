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

"""

Inception + BN, suitable for images with around 224 x 224

Reference:

Sergey Ioffe and Christian Szegedy. Batch normalization: Accelerating deep
network training by reducing internal covariate shift. arXiv preprint
arXiv:1502.03167, 2015.

"""
import mxnet as mx

eps = 1e-10 + 1e-5
bn_mom = 0.9
fix_gamma = False


def ConvFactory(data, num_filter, kernel, stride=(1,1), pad=(0, 0), name=None, suffix='', attr={}):
    conv = mx.symbol.Convolution(data=data, num_filter=num_filter, kernel=kernel, stride=stride, pad=pad, name='conv_%s%s' %(name, suffix))
    bn = mx.symbol.BatchNorm(data=conv, fix_gamma=fix_gamma, eps=eps, momentum=bn_mom, name='bn_%s%s' %(name, suffix))
    act = mx.symbol.Activation(data=bn, act_type='relu', name='relu_%s%s' %(name, suffix), attr=attr)
    return act

def InceptionFactoryA(data, num_1x1, num_3x3red, num_3x3, num_d3x3red, num_d3x3, pool, proj, name):
    # 1x1
    c1x1 = ConvFactory(data=data, num_filter=num_1x1, kernel=(1, 1), name=('%s_1x1' % name))
    # 3x3 reduce + 3x3
    c3x3r = ConvFactory(data=data, num_filter=num_3x3red, kernel=(1, 1), name=('%s_3x3' % name), suffix='_reduce')
    c3x3 = ConvFactory(data=c3x3r, num_filter=num_3x3, kernel=(3, 3), pad=(1, 1), name=('%s_3x3' % name))
    # double 3x3 reduce + double 3x3
    cd3x3r = ConvFactory(data=data, num_filter=num_d3x3red, kernel=(1, 1), name=('%s_double_3x3' % name), suffix='_reduce')
    cd3x3 = ConvFactory(data=cd3x3r, num_filter=num_d3x3, kernel=(3, 3), pad=(1, 1), name=('%s_double_3x3_0' % name))
    cd3x3 = ConvFactory(data=cd3x3, num_filter=num_d3x3, kernel=(3, 3), pad=(1, 1), name=('%s_double_3x3_1' % name))
    # pool + proj
    pooling = mx.symbol.Pooling(data=data, kernel=(3, 3), stride=(1, 1), pad=(1, 1), pool_type=pool, name=('%s_pool_%s_pool' % (pool, name)))
    cproj = ConvFactory(data=pooling, num_filter=proj, kernel=(1, 1), name=('%s_proj' %  name))
    # concat
    concat = mx.symbol.Concat(*[c1x1, c3x3, cd3x3, cproj], name='ch_concat_%s_chconcat' % name)
    return concat

def InceptionFactoryB(data, num_3x3red, num_3x3, num_d3x3red, num_d3x3, name):
    # 3x3 reduce + 3x3
    c3x3r = ConvFactory(data=data, num_filter=num_3x3red, kernel=(1, 1), name=('%s_3x3' % name), suffix='_reduce')
    c3x3 = ConvFactory(data=c3x3r, num_filter=num_3x3, kernel=(3, 3), pad=(1, 1), stride=(2, 2), name=('%s_3x3' % name))
    # double 3x3 reduce + double 3x3
    cd3x3r = ConvFactory(data=data, num_filter=num_d3x3red, kernel=(1, 1),  name=('%s_double_3x3' % name), suffix='_reduce')
    cd3x3 = ConvFactory(data=cd3x3r, num_filter=num_d3x3, kernel=(3, 3), pad=(1, 1), stride=(1, 1), name=('%s_double_3x3_0' % name))
    cd3x3 = ConvFactory(data=cd3x3, num_filter=num_d3x3, kernel=(3, 3), pad=(1, 1), stride=(2, 2), name=('%s_double_3x3_1' % name))
    # pool + proj
    pooling = mx.symbol.Pooling(data=data, kernel=(3, 3), stride=(2, 2), pad=(1, 1), pool_type="max", name=('max_pool_%s_pool' % name))
    # concat
    concat = mx.symbol.Concat(*[c3x3, cd3x3, pooling], name='ch_concat_%s_chconcat' % name)
    return concat

# A Simple Downsampling Factory
def DownsampleFactory(data, ch_3x3, name, attr):
    # conv 3x3
    conv = ConvFactory(data=data, name=name+'_conv',kernel=(3, 3), stride=(2, 2), num_filter=ch_3x3, pad=(1, 1), attr=attr)
    # pool
    pool = mx.symbol.Pooling(data=data, name=name+'_pool',kernel=(3, 3), stride=(2, 2), pad=(1, 1), pool_type='max', attr=attr)
    # concat
    concat = mx.symbol.Concat(*[conv, pool], name=name+'_ch_concat')
    return concat

# A Simple module
def SimpleFactory(data, ch_1x1, ch_3x3, name, attr):
    # 1x1
    conv1x1 = ConvFactory(data=data, name=name+'_1x1', kernel=(1, 1), pad=(0, 0), num_filter=ch_1x1, attr=attr)
    # 3x3
    conv3x3 = ConvFactory(data=data, name=name+'_3x3', kernel=(3, 3), pad=(1, 1), num_filter=ch_3x3, attr=attr)
    #concat
    concat = mx.symbol.Concat(*[conv1x1, conv3x3], name=name+'_ch_concat')
    return concat


def get_symbol(num_classes, image_shape, **kwargs):
    image_shape = [int(l) for l in image_shape.split(',')]
    (nchannel, height, width) = image_shape
    # attr = {'force_mirroring': 'true'}
    attr = {}

    # data
    data = mx.symbol.Variable(name="data")
    if height <= 28:
        # a simper version
        conv1 = ConvFactory(data=data, kernel=(3,3), pad=(1,1), name="1", num_filter=96, attr=attr)
        in3a = SimpleFactory(conv1, 32, 32, 'in3a', attr)
        in3b = SimpleFactory(in3a, 32, 48, 'in3b', attr)
        in3c = DownsampleFactory(in3b, 80, 'in3c', attr)
        in4a = SimpleFactory(in3c, 112, 48, 'in4a', attr)
        in4b = SimpleFactory(in4a, 96, 64, 'in4b', attr)
        in4c = SimpleFactory(in4b, 80, 80, 'in4c', attr)
        in4d = SimpleFactory(in4c, 48, 96, 'in4d', attr)
        in4e = DownsampleFactory(in4d, 96, 'in4e', attr)
        in5a = SimpleFactory(in4e, 176, 160, 'in5a', attr)
        in5b = SimpleFactory(in5a, 176, 160, 'in5b', attr)
        pool = mx.symbol.Pooling(data=in5b, pool_type="avg", kernel=(7,7), name="global_pool", attr=attr)
    else:
        # stage 1
        conv1 = ConvFactory(data=data, num_filter=64, kernel=(7, 7), stride=(2, 2), pad=(3, 3), name='1')
        pool1 = mx.symbol.Pooling(data=conv1, kernel=(3, 3), stride=(2, 2), name='pool_1', pool_type='max')
        # stage 2
        conv2red = ConvFactory(data=pool1, num_filter=64, kernel=(1, 1), stride=(1, 1), name='2_red')
        conv2 = ConvFactory(data=conv2red, num_filter=192, kernel=(3, 3), stride=(1, 1), pad=(1, 1), name='2')
        pool2 = mx.symbol.Pooling(data=conv2, kernel=(3, 3), stride=(2, 2), name='pool_2', pool_type='max')
        # stage 2
        in3a = InceptionFactoryA(pool2, 64, 64, 64, 64, 96, "avg", 32, '3a')
        in3b = InceptionFactoryA(in3a, 64, 64, 96, 64, 96, "avg", 64, '3b')
        in3c = InceptionFactoryB(in3b, 128, 160, 64, 96, '3c')
        # stage 3
        in4a = InceptionFactoryA(in3c, 224, 64, 96, 96, 128, "avg", 128, '4a')
        in4b = InceptionFactoryA(in4a, 192, 96, 128, 96, 128, "avg", 128, '4b')
        in4c = InceptionFactoryA(in4b, 160, 128, 160, 128, 160, "avg", 128, '4c')
        in4d = InceptionFactoryA(in4c, 96, 128, 192, 160, 192, "avg", 128, '4d')
        in4e = InceptionFactoryB(in4d, 128, 192, 192, 256, '4e')
        # stage 4
        in5a = InceptionFactoryA(in4e, 352, 192, 320, 160, 224, "avg", 128, '5a')
        in5b = InceptionFactoryA(in5a, 352, 192, 320, 192, 224, "max", 128, '5b')
        # global avg pooling
        pool = mx.symbol.Pooling(data=in5b, kernel=(7, 7), stride=(1, 1), name="global_pool", pool_type='avg')

    # linear classifier
    flatten = mx.symbol.Flatten(data=pool)
    fc1 = mx.symbol.FullyConnected(data=flatten, num_hidden=num_classes)
    softmax = mx.symbol.SoftmaxOutput(data=fc1, name='softmax')
    return softmax
