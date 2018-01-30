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
Inverted Residuals and Linear Bottlenecks:
Mobile Networks for Classification, Detection and Segmentation
https://arxiv.org/abs/1801.04381
'''
__author__ = 'dwSun'
__date__ = '18/1/31'

import mxnet as mx

from mxnet.gluon.model_zoo.vision.mobilenet import MobileNetV2


__all__ = ['MobileNetV2', 'get_symbol']


def get_symbol(num_classes=1000, multiplier=1.0, ctx=mx.cpu(), **kwargs):
    r"""MobileNetV2 model from the
    `"Inverted Residuals and Linear Bottlenecks:
      Mobile Networks for  Classification, Detection and Segmentation"
    <https://arxiv.org/abs/1801.04381>`_ paper.

    Parameters
    ----------
    num_classes : int, default 1000
        Number of classes for the output layer.
    multiplier : float, default 1.0
        The width multiplier for controling the model size. The actual number of channels
        is equal to the original channel size multiplied by this multiplier.
    ctx : Context, default CPU
        The context in which to initialize the model weights.
    """
    net = MobileNetV2(multiplier=multiplier, classes=num_classes, **kwargs)
    net.initialize(ctx=ctx, init=mx.init.Xavier())
    net.hybridize()

    data = mx.sym.var('data')
    out = net(data)
    sym = mx.sym.SoftmaxOutput(out, name='softmax')
    return sym


def plot_net():
    """
    Visualize the network.
    """
    sym = get_symbol(1000, prefix='mob_')

    # plot network graph
    mx.viz.plot_network(sym, shape={'data': (8, 3, 224, 224)},
                        node_attrs={'shape': 'oval', 'fixedsize': 'fasl==false'}).view()


if __name__ == '__main__':
    plot_net()
