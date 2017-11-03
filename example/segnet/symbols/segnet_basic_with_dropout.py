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

import mxnet as mx

def conv_deconv(input_data, num_classes, input_height=360, input_width=480):
    """
    get encode and decode layers
    """
    # norm
    norm_data = mx.symbol.LRN(data=input_data, alpha=0.0001, beta=0.75, nsize=5, name='lrn_norm')
    # group 1
    conv1 = mx.symbol.Convolution(data=norm_data, kernel=(7, 7), pad=(3, 3), num_filter=64,
                                  no_bias=True, name='conv1')
    bn_conv1 = mx.symbol.BatchNorm(data=conv1, name='bn_conv1')
    relu1 = mx.symbol.Activation(data=bn_conv1, act_type='relu', name='relu1')
    pool1 = mx.symbol.PoolingMask(data=relu1, kernel=(2, 2), stride=(2, 2), name='pool1')
    # group 2
    conv2 = mx.symbol.Convolution(data=pool1[0], kernel=(7, 7), pad=(3, 3), num_filter=64,
                                  no_bias=True, name='conv2')
    bn_conv2 = mx.symbol.BatchNorm(data=conv2, name='bn_conv2')
    relu2 = mx.symbol.Activation(data=bn_conv2, act_type='relu', name='relu2')
    pool2 = mx.symbol.PoolingMask(data=relu2, kernel=(2, 2), stride=(2, 2), name='pool2')
    # group 3
    conv3 = mx.symbol.Convolution(data=pool2[0], kernel=(7, 7), pad=(3, 3), num_filter=64,
                                  no_bias=True, name='conv3')
    bn_conv3 = mx.symbol.BatchNorm(data=conv3, name='bn_conv3')
    relu3 = mx.symbol.Activation(data=bn_conv3, act_type='relu', name='relu3')
    pool3 = mx.symbol.PoolingMask(data=relu3, kernel=(2, 2), stride=(2, 2), name='pool3')
    # group 4
    conv4 = mx.symbol.Convolution(data=pool3[0], kernel=(7, 7), pad=(3, 3), num_filter=64,
                                  no_bias=True, name='conv4')
    bn_conv4 = mx.symbol.BatchNorm(data=conv4, name='bn_conv4')
    relu4 = mx.symbol.Activation(data=bn_conv4, act_type='relu', name='relu4')
    pool4 = mx.symbol.PoolingMask(data=relu4, kernel=(2, 2), stride=(2, 2), name='pool4')
    # deconve
    # group 4
    pool4_d = mx.symbol.UpSamplingMask(data=pool4[0], mask=pool4[1],
                                       out_shape=(input_height/8, input_width/8), name="pool4_d")
    conv4_d = mx.symbol.Convolution(data=pool4_d, kernel=(7, 7), pad=(3, 3), num_filter=64,
                                    no_bias=True, name='conv4_d')
    bn_conv4_d = mx.symbol.BatchNorm(data=conv4_d, name='bn_conv4_d')
    drop1 = mx.symbol.Dropout(data=bn_conv4_d, p=0.5, name='drop1')
    # group 3
    pool3_d = mx.symbol.UpSamplingMask(data=drop1, mask=pool3[1],
                                       out_shape=(input_height/4, input_width/4), name="pool3_d")
    conv3_d = mx.symbol.Convolution(data=pool3_d, kernel=(7, 7), pad=(3, 3), num_filter=64,
                                    no_bias=True, name='conv3_d')
    bn_conv3_d = mx.symbol.BatchNorm(data=conv3_d, name='bn_conv3_d')
    drop2 = mx.symbol.Dropout(data=bn_conv3_d, p=0.5, name='drop2')
    # group 2
    pool2_d = mx.symbol.UpSamplingMask(data=drop2, mask=pool2[1],
                                       out_shape=(input_height/2, input_width/2), name="pool2_d")
    conv2_d = mx.symbol.Convolution(data=pool2_d, kernel=(7, 7), pad=(3, 3), num_filter=64,
                                    no_bias=True, name='conv2_d')
    bn_conv2_d = mx.symbol.BatchNorm(data=conv2_d, name='bn_conv2_d')
    # group 1
    pool1_d = mx.symbol.UpSamplingMask(data=bn_conv2_d, mask=pool1[1],
                                       out_shape=(input_height, input_width), name="pool1_d")
    conv1_d = mx.symbol.Convolution(data=pool1_d, kernel=(7, 7), pad=(3, 3), num_filter=64,
                                    no_bias=True, name='conv1_d')
    bn_conv1_d = mx.symbol.BatchNorm(data=conv1_d, name='bn_conv1_d')
    # classfiier
    conv_classfiier = mx.symbol.Convolution(data=bn_conv1_d, kernel=(1, 1),
                                            num_filter=num_classes, name='conv_classfiier')
    return conv_classfiier

def get_symbol(num_classes, natural_balance=False):
    """
    get segnet_basic network
    """
    data = mx.symbol.Variable(name='data')
    network = conv_deconv(data, num_classes)
    if natural_balance:
        weight = mx.symbol.Variable(name="softmax_weight", lr_mult=0)
        label = mx.symbol.Variable(name='softmax_label', lr_mult=0)
        softmax = mx.symbol.softmax(data=network, axis=1, name='softmax')
        label = mx.symbol.one_hot(indices=label, depth=num_classes)
        label = mx.symbol.swapaxes(label, 1, 3)
        label = mx.symbol.swapaxes(label, 2, 3)
        cross_entropy = mx.symbol.broadcast_mul(mx.symbol.log(softmax), label)
        loss = mx.sym.MakeLoss(-cross_entropy*weight, grad_scale=1.0/(360*480))
        return loss
    else:
        softmax = mx.symbol.SoftmaxOutput(data=network, multi_output=True, use_ignore=True,
                                          ignore_label=11, name='softmax')
        return softmax
