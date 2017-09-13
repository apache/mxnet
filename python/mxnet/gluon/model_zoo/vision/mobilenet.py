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
# pylint: disable= arguments-differ
"""MobileNet, implemented in Gluon."""
__all__ = ['MobileNet', 'mobilenet']

from ....context import cpu
from ...block import HybridBlock
from ... import nn

# Helpers
def _make_conv(channels=1, kernel=(1, 1), stride=(1, 1), pad=(0, 0), num_group=1):
    out = nn.HybridSequential(prefix='')
    out.add(nn.Conv2D(channels, kernel, stride, pad, groups=num_group, use_bias=False))
    out.add(nn.BatchNorm(scale=False))
    out.add(nn.Activation('relu'))
    return out

# Net
class MobileNet(HybridBlock):
    r"""MobileNet model from the
    `"MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications"
    <https://arxiv.org/abs/1704.04861>`_ paper.

    Parameters
    ----------
    classes : int, default 1000
        Number of classes for the output layer.
    """
    def __init__(self, classes=1000, **kwargs):
        super(MobileNet, self).__init__(**kwargs)
        with self.name_scope():
            self.features = nn.HybridSequential(prefix='')
            with self.features.name_scope():
                self.features.add(_make_conv(channels=32, kernel=3, pad=1, stride=2))
                self.features.add(_make_conv(num_group=32, channels=32, kernel=3, pad=1))
                self.features.add(_make_conv(channels=64))
                self.features.add(_make_conv(num_group=64, channels=64, kernel=3, pad=1, stride=2))
                self.features.add(_make_conv(channels=128))
                self.features.add(_make_conv(num_group=128, channels=128, kernel=3, pad=1))
                self.features.add(_make_conv(channels=128))
                self.features.add(_make_conv(num_group=128, channels=128,
                                             kernel=3, pad=1, stride=2))
                self.features.add(_make_conv(channels=256))
                self.features.add(_make_conv(num_group=256, channels=256, kernel=3, pad=1))
                self.features.add(_make_conv(channels=256))
                self.features.add(_make_conv(num_group=256, channels=256,
                                             kernel=3, pad=1, stride=2))
                self.features.add(_make_conv(channels=512))
                self.features.add(_make_conv(num_group=512, channels=512, kernel=3, pad=1))
                self.features.add(_make_conv(channels=512))
                self.features.add(_make_conv(num_group=512, channels=512, kernel=3, pad=1))
                self.features.add(_make_conv(channels=512))
                self.features.add(_make_conv(num_group=512, channels=512, kernel=3, pad=1))
                self.features.add(_make_conv(channels=512))
                self.features.add(_make_conv(num_group=512, channels=512, kernel=3, pad=1))
                self.features.add(_make_conv(channels=512))
                self.features.add(_make_conv(num_group=512, channels=512, kernel=3, pad=1))
                self.features.add(_make_conv(channels=512))
                self.features.add(_make_conv(num_group=512, channels=512,
                                             kernel=3, pad=1, stride=2))
                self.features.add(_make_conv(channels=1024))
                self.features.add(_make_conv(num_group=1024, channels=1024, kernel=3, pad=1))
                self.features.add(_make_conv(channels=1024))
                self.features.add(nn.AvgPool2D(7, strides=1))
                self.features.add(nn.Flatten())

            self.classifier = nn.HybridSequential(prefix='')
            with self.classifier.name_scope():
                self.classifier.add(nn.Dense(classes))

    def hybrid_forward(self, F, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# Constructor
def mobilenet(pretrained=False, ctx=cpu(), **kwargs):
    r"""MobileNet model from the
    `"MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications"
    <https://arxiv.org/abs/1704.04861>`_ paper.

    Parameters
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    """
    net = MobileNet(**kwargs)
    if pretrained:
        from ..model_store import get_model_file
        net.load_params(get_model_file('mobilenet'), ctx=ctx)
    return net
