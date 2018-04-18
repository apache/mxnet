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


# coding: utf-8

import mxnet as mx
import numpy as np


def Conv(data, num_filter, kernel=(5, 5), pad=(2, 2), stride=(2, 2)):
    sym = mx.sym.Convolution(data, num_filter=num_filter, kernel=kernel, stride=stride, pad=pad, no_bias=False)
    sym = mx.sym.BatchNorm(sym, fix_gamma=False)
    sym = mx.sym.LeakyReLU(sym, act_type="leaky")
    return sym


def Deconv(data, num_filter, im_hw, kernel=(7, 7), pad=(2, 2), stride=(2, 2), crop=True, out=False):
    sym = mx.sym.Deconvolution(data, num_filter=num_filter, kernel=kernel, stride=stride, pad=pad, no_bias=True)
    if crop:
        sym = mx.sym.Crop(sym, offset=(1, 1), h_w=im_hw, num_args=1)
    sym = mx.sym.BatchNorm(sym, fix_gamma=False)
    if out == False:
        sym = mx.sym.LeakyReLU(sym, act_type="leaky")
    else:
        sym = mx.sym.Activation(sym, act_type="tanh")
    return sym

def get_generator(prefix, im_hw):
    data = mx.sym.Variable("%s_data" % prefix)
    conv1 = Conv(data, 64) # 192
    conv1_1 = Conv(conv1, 48, kernel=(3, 3), pad=(1, 1), stride=(1, 1))
    conv2 = Conv(conv1_1, 128) # 96
    conv2_1 = Conv(conv2, 96, kernel=(3, 3), pad=(1, 1), stride=(1, 1))
    conv3 = Conv(conv2_1, 256) # 48
    conv3_1 = Conv(conv3, 192, kernel=(3, 3), pad=(1, 1), stride=(1, 1))
    deconv1 = Deconv(conv3_1, 128, (int(im_hw[0] / 4), int(im_hw[1] / 4))) + conv2
    conv4_1 = Conv(deconv1, 160, kernel=(3, 3), pad=(1, 1), stride=(1, 1))
    deconv2 = Deconv(conv4_1, 64, (int(im_hw[0] / 2), int(im_hw[1] / 2))) + conv1
    conv5_1 = Conv(deconv2, 96, kernel=(3, 3), pad=(1, 1), stride=(1, 1))
    deconv3 = Deconv(conv5_1, 3, im_hw, kernel=(8,  8), pad=(3, 3), out=True, crop=False)
    raw_out = (deconv3 * 128) + 128
    norm = mx.sym.SliceChannel(raw_out, num_outputs=3)
    r_ch = norm[0] - 123.68
    g_ch = norm[1] - 116.779
    b_ch = norm[2] - 103.939
    norm_out = 0.4 * mx.sym.Concat(*[r_ch, g_ch, b_ch]) + 0.6 * data
    return norm_out

def get_module(prefix, dshape, ctx, is_train=True):
    sym = get_generator(prefix, dshape[-2:])
    mod = mx.mod.Module(symbol=sym,
                        data_names=("%s_data" % prefix,),
                        label_names=None,
                        context=ctx)
    if is_train:
        mod.bind(data_shapes=[("%s_data" % prefix, dshape)], for_training=True, inputs_need_grad=True)
    else:
        mod.bind(data_shapes=[("%s_data" % prefix, dshape)], for_training=False, inputs_need_grad=False)
    mod.init_params(initializer=mx.init.Xavier(magnitude=2.))
    return mod




