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

def vgg_conv_deconv(input_data, num_classes, fix_gamma=False, eps=0.00002,
                    bn_mom=0.9, input_height=360, input_width=480):
    """
    get vgg encode and decode layers
    """
    # group 1
    conv1_1 = mx.symbol.Convolution(data=input_data, kernel=(3, 3), pad=(1, 1), num_filter=64,
                                    name="conv1_1")
    bn_conv1_1 = mx.symbol.BatchNorm(data=conv1_1, fix_gamma=fix_gamma, eps=eps, momentum=bn_mom,
                                     name='bn_conv1_1')
    relu1_1 = mx.symbol.Activation(data=bn_conv1_1, act_type="relu", name="relu1_1")
    conv1_2 = mx.symbol.Convolution(data=relu1_1, kernel=(3, 3), pad=(1, 1), num_filter=64,
                                    name="conv1_2")
    bn_conv1_2 = mx.symbol.BatchNorm(data=conv1_2, fix_gamma=fix_gamma, eps=eps, momentum=bn_mom,
                                     name='bn_conv1_2')
    relu1_2 = mx.symbol.Activation(data=bn_conv1_2, act_type="relu", name="relu1_2")
    pool1 = mx.symbol.PoolingMask(data=relu1_2, kernel=(2, 2), stride=(2, 2), name="pool1")
    # group 2
    conv2_1 = mx.symbol.Convolution(data=pool1[0], kernel=(3, 3), pad=(1, 1), num_filter=128,
                                    name="conv2_1")
    bn_conv2_1 = mx.symbol.BatchNorm(data=conv2_1, fix_gamma=fix_gamma, eps=eps, momentum=bn_mom,
                                     name='bn_conv2_1')
    relu2_1 = mx.symbol.Activation(data=bn_conv2_1, act_type="relu", name="relu2_1")
    conv2_2 = mx.symbol.Convolution(data=relu2_1, kernel=(3, 3), pad=(1, 1), num_filter=128,
                                    name="conv2_2")
    bn_conv2_2 = mx.symbol.BatchNorm(data=conv2_2, fix_gamma=fix_gamma, eps=eps, momentum=bn_mom,
                                     name='bn_conv2_2')
    relu2_2 = mx.symbol.Activation(data=bn_conv2_2, act_type="relu", name="relu2_2")
    pool2 = mx.symbol.PoolingMask(data=relu2_2, kernel=(2, 2), stride=(2, 2), name="pool2")
    # group 3
    conv3_1 = mx.symbol.Convolution(data=pool2[0], kernel=(3, 3), pad=(1, 1), num_filter=256,
                                    name="conv3_1")
    bn_conv3_1 = mx.symbol.BatchNorm(data=conv3_1, fix_gamma=fix_gamma, eps=eps, momentum=bn_mom,
                                     name='bn_conv3_1')
    relu3_1 = mx.symbol.Activation(data=bn_conv3_1, act_type="relu", name="relu3_1")
    conv3_2 = mx.symbol.Convolution(data=relu3_1, kernel=(3, 3), pad=(1, 1), num_filter=256,
                                    name="conv3_2")
    bn_conv3_2 = mx.symbol.BatchNorm(data=conv3_2, fix_gamma=fix_gamma, eps=eps, momentum=bn_mom,
                                     name='bn_conv3_2')
    relu3_2 = mx.symbol.Activation(data=bn_conv3_2, act_type="relu", name="relu3_2")
    conv3_3 = mx.symbol.Convolution(data=relu3_2, kernel=(3, 3), pad=(1, 1), num_filter=256,
                                    name="conv3_3")
    bn_conv3_3 = mx.symbol.BatchNorm(data=conv3_3, fix_gamma=fix_gamma, eps=eps, momentum=bn_mom,
                                     name='bn_conv3_3')
    relu3_3 = mx.symbol.Activation(data=bn_conv3_3, act_type="relu", name="relu3_3")
    pool3 = mx.symbol.PoolingMask(data=relu3_3, kernel=(2, 2), stride=(2, 2), name="pool3")
    # group 4
    conv4_1 = mx.symbol.Convolution(data=pool3[0], kernel=(3, 3), pad=(1, 1), num_filter=512,
                                    name="conv4_1")
    bn_conv4_1 = mx.symbol.BatchNorm(data=conv4_1, fix_gamma=fix_gamma, eps=eps, momentum=bn_mom,
                                     name='bn_conv4_1')
    relu4_1 = mx.symbol.Activation(data=bn_conv4_1, act_type="relu", name="relu4_1")
    conv4_2 = mx.symbol.Convolution(data=relu4_1, kernel=(3, 3), pad=(1, 1), num_filter=512,
                                    name="conv4_2")
    bn_conv4_2 = mx.symbol.BatchNorm(data=conv4_2, fix_gamma=fix_gamma, eps=eps, momentum=bn_mom,
                                     name='bn_conv4_2')
    relu4_2 = mx.symbol.Activation(data=bn_conv4_2, act_type="relu", name="relu4_2")
    conv4_3 = mx.symbol.Convolution(data=relu4_2, kernel=(3, 3), pad=(1, 1), num_filter=512,
                                    name="conv4_3")
    bn_conv4_3 = mx.symbol.BatchNorm(data=conv4_3, fix_gamma=fix_gamma, eps=eps, momentum=bn_mom,
                                     name='bn_conv4_3')
    relu4_3 = mx.symbol.Activation(data=bn_conv4_3, act_type="relu", name="relu4_3")
    pool4 = mx.symbol.PoolingMask(data=relu4_3, kernel=(2, 2), stride=(2, 2), name="pool4")
    # group 5
    conv5_1 = mx.symbol.Convolution(data=pool4[0], kernel=(3, 3), pad=(1, 1), num_filter=512,
                                    name="conv5_1")
    bn_conv5_1 = mx.symbol.BatchNorm(data=conv5_1, fix_gamma=fix_gamma, eps=eps, momentum=bn_mom,
                                     name='bn_conv5_1')
    relu5_1 = mx.symbol.Activation(data=bn_conv5_1, act_type="relu", name="relu5_1")
    conv5_2 = mx.symbol.Convolution(data=relu5_1, kernel=(3, 3), pad=(1, 1), num_filter=512,
                                    name="conv5_2")
    bn_conv5_2 = mx.symbol.BatchNorm(data=conv5_2, fix_gamma=fix_gamma, eps=eps, momentum=bn_mom,
                                     name='bn_conv5_2')
    relu5_2 = mx.symbol.Activation(data=bn_conv5_2, act_type="relu", name="relu5_2")
    conv5_3 = mx.symbol.Convolution(data=relu5_2, kernel=(3, 3), pad=(1, 1), num_filter=512,
                                    name="conv5_3")
    bn_conv5_3 = mx.symbol.BatchNorm(data=conv5_3, fix_gamma=fix_gamma, eps=eps, momentum=bn_mom,
                                     name='bn_conv5_3')
    relu5_3 = mx.symbol.Activation(data=bn_conv5_3, act_type="relu", name="relu5_3")
    pool5 = mx.symbol.PoolingMask(data=relu5_3, kernel=(2, 2), stride=(2, 2), name="pool5")
    # deconve
    # group 5
    pool5_d = mx.symbol.UpSamplingMask(data=pool5[0], mask=pool5[1],
                                       out_shape=(input_height/16, input_width/16), name="pool5_d")
    conv5_3_d = mx.symbol.Convolution(data=pool5_d, kernel=(3, 3), pad=(1, 1), num_filter=512,
                                      name="conv5_3_d")
    bn_conv5_3_d = mx.symbol.BatchNorm(data=conv5_3_d, fix_gamma=fix_gamma, eps=eps,
                                       momentum=bn_mom, name='bn_conv5_3_d')
    relu5_3_d = mx.symbol.Activation(data=bn_conv5_3_d, act_type="relu", name="relu5_3_d")
    conv5_2_d = mx.symbol.Convolution(data=relu5_3_d, kernel=(3, 3), pad=(1, 1), num_filter=512,
                                      name="conv5_2_d")
    bn_conv5_2_d = mx.symbol.BatchNorm(data=conv5_2_d, fix_gamma=fix_gamma, eps=eps,
                                       momentum=bn_mom, name='bn_conv5_2_d')
    relu5_2_d = mx.symbol.Activation(data=bn_conv5_2_d, act_type="relu", name="relu5_2_d")
    conv5_1_d = mx.symbol.Convolution(data=relu5_2_d, kernel=(3, 3), pad=(1, 1), num_filter=512,
                                      name="conv5_1_d")
    bn_conv5_1_d = mx.symbol.BatchNorm(data=conv5_1_d, fix_gamma=fix_gamma, eps=eps,
                                       momentum=bn_mom, name='bn_conv5_1_d')
    relu5_1_d = mx.symbol.Activation(data=bn_conv5_1_d, act_type="relu", name="relu5_1_d")
    # group 4
    pool4_d = mx.symbol.UpSamplingMask(data=relu5_1_d, mask=pool4[1],
                                       out_shape=(input_height/8, input_width/8), name="pool4_d")
    conv4_3_d = mx.symbol.Convolution(data=pool4_d, kernel=(3, 3), pad=(1, 1), num_filter=512,
                                      name="conv4_3_d")
    bn_conv4_3_d = mx.symbol.BatchNorm(data=conv4_3_d, fix_gamma=fix_gamma, eps=eps,
                                       momentum=bn_mom, name='bn_conv4_3_d')
    relu4_3_d = mx.symbol.Activation(data=bn_conv4_3_d, act_type="relu", name="relu4_3_d")
    conv4_2_d = mx.symbol.Convolution(data=relu4_3_d, kernel=(3, 3), pad=(1, 1), num_filter=512,
                                      name="conv4_2_d")
    bn_conv4_2_d = mx.symbol.BatchNorm(data=conv4_2_d, fix_gamma=fix_gamma, eps=eps,
                                       momentum=bn_mom, name='bn_conv4_2_d')
    relu4_2_d = mx.symbol.Activation(data=bn_conv4_2_d, act_type="relu", name="relu4_2_d")
    conv4_1_d = mx.symbol.Convolution(data=relu4_2_d, kernel=(3, 3), pad=(1, 1), num_filter=256,
                                      name="conv4_1_d")
    bn_conv4_1_d = mx.symbol.BatchNorm(data=conv4_1_d, fix_gamma=fix_gamma, eps=eps,
                                       momentum=bn_mom, name='bn_conv4_1_d')
    relu4_1_d = mx.symbol.Activation(data=bn_conv4_1_d, act_type="relu", name="relu4_1_d")
    # group 3
    pool3_d = mx.symbol.UpSamplingMask(data=relu4_1_d, mask=pool3[1],
                                       out_shape=(input_height/4, input_width/4), name="pool3_d")
    conv3_3_d = mx.symbol.Convolution(data=pool3_d, kernel=(3, 3), pad=(1, 1), num_filter=256,
                                      name="conv3_3_d")
    bn_conv3_3_d = mx.symbol.BatchNorm(data=conv3_3_d, fix_gamma=fix_gamma, eps=eps,
                                       momentum=bn_mom, name='bn_conv3_3_d')
    relu3_3_d = mx.symbol.Activation(data=bn_conv3_3_d, act_type="relu", name="relu3_3_d")
    conv3_2_d = mx.symbol.Convolution(data=relu3_3_d, kernel=(3, 3), pad=(1, 1), num_filter=256,
                                      name="conv3_2_d")
    bn_conv3_2_d = mx.symbol.BatchNorm(data=conv3_2_d, fix_gamma=fix_gamma, eps=eps,
                                       momentum=bn_mom, name='bn_conv3_2_d')
    relu3_2_d = mx.symbol.Activation(data=bn_conv3_2_d, act_type="relu", name="relu3_2_d")
    conv3_1_d = mx.symbol.Convolution(data=relu3_2_d, kernel=(3, 3), pad=(1, 1), num_filter=128,
                                      name="conv3_1_d")
    bn_conv3_1_d = mx.symbol.BatchNorm(data=conv3_1_d, fix_gamma=fix_gamma, eps=eps,
                                       momentum=bn_mom, name='bn_conv3_1_d')
    relu3_1_d = mx.symbol.Activation(data=bn_conv3_1_d, act_type="relu", name="relu3_1_d")
    # group 2
    pool2_d = mx.symbol.UpSamplingMask(data=relu3_1_d, mask=pool2[1],
                                       out_shape=(input_height/2, input_width/2), name="pool2_d")
    conv2_2_d = mx.symbol.Convolution(data=pool2_d, kernel=(3, 3), pad=(1, 1), num_filter=128,
                                      name="conv2_2_d")
    bn_conv2_2_d = mx.symbol.BatchNorm(data=conv2_2_d, fix_gamma=fix_gamma, eps=eps,
                                       momentum=bn_mom, name='bn_conv2_2_d')
    relu2_2_d = mx.symbol.Activation(data=bn_conv2_2_d, act_type="relu", name="relu2_2_d")
    conv2_1_d = mx.symbol.Convolution(data=relu2_2_d, kernel=(3, 3), pad=(1, 1), num_filter=64,
                                      name="conv2_1_d")
    bn_conv2_1_d = mx.symbol.BatchNorm(data=conv2_1_d, fix_gamma=fix_gamma, eps=eps,
                                       momentum=bn_mom, name='bn_conv2_1_d')
    relu2_1_d = mx.symbol.Activation(data=bn_conv2_1_d, act_type="relu", name="relu2_1_d")
    # group 1
    pool1_d = mx.symbol.UpSamplingMask(data=relu2_1_d, mask=pool1[1],
                                       out_shape=(input_height, input_width), name="pool1_d")
    conv1_2_d = mx.symbol.Convolution(data=pool1_d, kernel=(3, 3), pad=(1, 1), num_filter=64,
                                      name="conv1_2_d")
    bn_conv1_2_d = mx.symbol.BatchNorm(data=conv1_2_d, fix_gamma=fix_gamma, eps=eps,
                                       momentum=bn_mom, name='bn_conv1_2_d')
    relu1_2_d = mx.symbol.Activation(data=bn_conv1_2_d, act_type="relu", name="relu1_2_d")
    conv1_1_d = mx.symbol.Convolution(data=relu1_2_d, kernel=(3, 3), pad=(1, 1), num_filter=num_classes,
                                      name="conv1_1_d")
    return conv1_1_d

def get_symbol(num_classes, natural_balance=False):
    """
    get segnet network
    """
    data = mx.symbol.Variable(name='data')
    network = vgg_conv_deconv(data, num_classes)
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
