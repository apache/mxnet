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
'''
MobileNetV2, implemented in Gluon.

Reference:
Inverted Residuals and Linear Bottlenecks: Mobile Networks for  Classification, Detection and Segmentation
https://arxiv.org/abs/1801.04381
'''
__author__ = 'dwSun'
__date__ = '18/1/29'

import mxnet as mx
import mxnet.gluon.nn as nn

__all__ = ['MobileNetV2', 'get_symbol']


class BottleNeck(nn.HybridBlock):
    def __init__(self, c_in, c_out, t,  s, **kwargs):
        super(BottleNeck, self).__init__(**kwargs)
        self.use_shortcut = s == 1 and c_in == c_out
        with self.name_scope():
            self.out = nn.HybridSequential()
            self.out.add(
                nn.Conv2D(c_in * t, 1, padding=0, use_bias=False),
                nn.BatchNorm(scale=True),
                nn.Activation('relu'),

                nn.Conv2D(c_in * t, 3, strides=s, padding=1, groups=c_in * t, use_bias=False),
                nn.BatchNorm(scale=True),
                nn.Activation('relu'),

                nn.Conv2D(c_out, 1, padding=0, use_bias=False),
                nn.BatchNorm(scale=True),
            )

    def hybrid_forward(self, F, x):
        out = self.out(x)
        if self.use_shortcut:
            out = F.elemwise_add(out, x)
        return out


class MobileNetV2(nn.HybridBlock):
    r"""MobileNetV2 model from the
    `"Inverted Residuals and Linear Bottlenecks: Mobile Networks for  Classification, Detection and Segmentation"
    <https://arxiv.org/abs/1801.04381>`_ paper.

    Parameters
    ----------
    num_classes : int, default 1000
        Number of classes for the output layer.
    w : float, default 1.0
        The width multiplier for controling the model size. The actual number of channels
        is equal to the original channel size multiplied by this multiplier.
    """
    def __init__(self, num_classes=1000, w=1.0, **kwargs):
        super(MobileNetV2, self).__init__(**kwargs)
        with self.name_scope():
            self.features = nn.HybridSequential(prefix='features_')
            with self.features.name_scope():
                self.features.add(
                    nn.Conv2D(int(32 * w), 3, strides=2, padding=1, use_bias=False),
                    nn.BatchNorm(scale=True),
                    nn.Activation('relu')
                )

                self.features.add(BottleNeck(c_in=int(32 * w), c_out=int(16 * w), t=1, s=1))

                self.features.add(BottleNeck(c_in=int(16 * w), c_out=int(24 * w), t=6, s=2))
                self.features.add(BottleNeck(c_in=int(24 * w), c_out=int(24 * w), t=6, s=1))

                self.features.add(BottleNeck(c_in=int(24 * w), c_out=int(32 * w), t=6, s=2))
                self.features.add(BottleNeck(c_in=int(32 * w), c_out=int(32 * w), t=6, s=1))
                self.features.add(BottleNeck(c_in=int(32 * w), c_out=int(32 * w), t=6, s=1))

                self.features.add(BottleNeck(c_in=int(32 * w), c_out=int(64 * w), t=6, s=2))
                self.features.add(BottleNeck(c_in=int(64 * w), c_out=int(64 * w), t=6, s=1))
                self.features.add(BottleNeck(c_in=int(64 * w), c_out=int(64 * w), t=6, s=1))
                self.features.add(BottleNeck(c_in=int(64 * w), c_out=int(64 * w), t=6, s=1))

                self.features.add(BottleNeck(c_in=int(64 * w), c_out=int(96 * w), t=6, s=1))
                self.features.add(BottleNeck(c_in=int(96 * w), c_out=int(96 * w), t=6, s=1))
                self.features.add(BottleNeck(c_in=int(96 * w), c_out=int(96 * w), t=6, s=1))

                self.features.add(BottleNeck(c_in=int(96 * w), c_out=int(160 * w), t=6, s=2))
                self.features.add(BottleNeck(c_in=int(160 * w), c_out=int(160 * w), t=6, s=1))
                self.features.add(BottleNeck(c_in=int(160 * w), c_out=int(160 * w), t=6, s=1))

                self.features.add(BottleNeck(c_in=int(160 * w), c_out=int(320 * w), t=6, s=1))

                last_channels = int(1280 * w) if w > 1.0 else 1280

                self.features.add(
                    nn.Conv2D(last_channels, 1, strides=1, padding=0, use_bias=False),
                    nn.BatchNorm(scale=True),
                    nn.Activation('relu'),
                )
                self.features.add(nn.GlobalAvgPool2D())

            self.output = nn.Conv2D(channels=num_classes, kernel_size=1, strides=1, padding=0, use_bias=False, prefix='pred_')
            self.flatten = nn.Flatten(prefix='flat_')

    def hybrid_forward(self, F, x):
        x = self.features(x)
        x = self.output(x)
        x = self.flatten(x)
        return x


def get_symbol(num_classes=1000, w=1.0, ctx=mx.cpu()):
    r"""MobileNetV2 model from the
    `"Inverted Residuals and Linear Bottlenecks: Mobile Networks for  Classification, Detection and Segmentation"
    <https://arxiv.org/abs/1801.04381>`_ paper.

    Parameters
    ----------
    num_classes : int, default 1000
        Number of classes for the output layer.
    w : float, default 1.0
        The width multiplier for controling the model size. The actual number of channels
        is equal to the original channel size multiplied by this multiplier.
    ctx : Context, default CPU
        The context in which to initialize the model weights.
    """
    net = MobileNetV2(num_classes, w)
    net.initialize(ctx=ctx, init=mx.init.Xavier())
    net.hybridize()

    data = mx.sym.var('data')
    out = net(data)
    sym = mx.sym.SoftmaxOutput(out, name='softmax')
    return sym


if __name__ == '__main__':
    net = MobileNetV2(1000, prefix='mob_')

    data = mx.sym.var('data')
    sym = net(data)

    # plot network graph
    mx.viz.plot_network(sym, shape={'data': (8, 3, 224, 224)}, node_attrs={'shape': 'oval', 'fixedsize': 'fasl==false'}).view()
