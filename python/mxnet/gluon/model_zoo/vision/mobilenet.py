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
"""MobileNet and MobileNetV2, implemented in Gluon."""
__all__ = ['MobileNet', 'MobileNetV2', 'mobilenet1_0', 'mobilenet_v2_1_0', 'mobilenet0_75',
           'mobilenet_v2_0_75', 'mobilenet0_5', 'mobilenet_v2_0_5', 'mobilenet0_25',
           'mobilenet_v2_0_25', 'get_mobilenet', 'get_mobilenet_v2']

__modify__ = 'dwSun'
__modified_date__ = '18/04/18'

import os

from ... import nn
from ....device import cpu
from ...block import HybridBlock
from .... import base, np
from ....util import use_np, wrap_ctx_to_device_func


# Helpers
@use_np
class RELU6(nn.HybridBlock):
    """Relu6 used in MobileNetV2."""

    def __init__(self, **kwargs):
        super(RELU6, self).__init__(**kwargs)

    def forward(self, x):
        return np.clip(x, 0, 6)


# pylint: disable= too-many-arguments
def _add_conv(out, channels=1, kernel=1, stride=1, pad=0,
              num_group=1, active=True, relu6=False):
    out.add(nn.Conv2D(channels, kernel, stride, pad, groups=num_group, use_bias=False))
    out.add(nn.BatchNorm(scale=True))
    if active:
        out.add(RELU6() if relu6 else nn.Activation('relu'))


def _add_conv_dw(out, dw_channels, channels, stride, relu6=False):
    _add_conv(out, channels=dw_channels, kernel=3, stride=stride,
              pad=1, num_group=dw_channels, relu6=relu6)
    _add_conv(out, channels=channels, relu6=relu6)


@use_np
class LinearBottleneck(nn.HybridBlock):
    r"""LinearBottleneck used in MobileNetV2 model from the
    `"Inverted Residuals and Linear Bottlenecks:
    Mobile Networks for Classification, Detection and Segmentation"
    <https://arxiv.org/abs/1801.04381>`_ paper.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    channels : int
        Number of output channels.
    t : int
        Layer expansion ratio.
    stride : int
        stride
    """

    def __init__(self, in_channels, channels, t, stride, **kwargs):
        super(LinearBottleneck, self).__init__(**kwargs)
        self.use_shortcut = stride == 1 and in_channels == channels
        self.out = nn.HybridSequential()

        _add_conv(self.out, in_channels * t, relu6=True)
        _add_conv(self.out, in_channels * t, kernel=3, stride=stride,
                  pad=1, num_group=in_channels * t, relu6=True)
        _add_conv(self.out, channels, active=False, relu6=True)

    def forward(self, x):
        out = self.out(x)
        if self.use_shortcut:
            out = np.add(out, x)
        return out


# Net
@use_np
class MobileNet(HybridBlock):
    r"""MobileNet model from the
    `"MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications"
    <https://arxiv.org/abs/1704.04861>`_ paper.

    Parameters
    ----------
    multiplier : float, default 1.0
        The width multiplier for controling the model size. Only multipliers that are no
        less than 0.25 are supported. The actual number of channels is equal to the original
        channel size multiplied by this multiplier.
    classes : int, default 1000
        Number of classes for the output layer.
    """

    def __init__(self, multiplier=1.0, classes=1000, **kwargs):
        super(MobileNet, self).__init__(**kwargs)
        self.features = nn.HybridSequential()
        _add_conv(self.features, channels=int(32 * multiplier), kernel=3, pad=1, stride=2)
        dw_channels = [int(x * multiplier) for x in [32, 64] + [128] * 2
                       + [256] * 2 + [512] * 6 + [1024]]
        channels = [int(x * multiplier) for x in [64] + [128] * 2 + [256] * 2
                    + [512] * 6 + [1024] * 2]
        strides = [1, 2] * 3 + [1] * 5 + [2, 1]
        for dwc, c, s in zip(dw_channels, channels, strides):
            _add_conv_dw(self.features, dw_channels=dwc, channels=c, stride=s)
        self.features.add(nn.GlobalAvgPool2D())
        self.features.add(nn.Flatten())

        self.output = nn.Dense(classes)

    def forward(self, x):
        x = self.features(x)
        x = self.output(x)
        return x


@use_np
class MobileNetV2(nn.HybridBlock):
    r"""MobileNetV2 model from the
    `"Inverted Residuals and Linear Bottlenecks:
    Mobile Networks for Classification, Detection and Segmentation"
    <https://arxiv.org/abs/1801.04381>`_ paper.

    Parameters
    ----------
    multiplier : float, default 1.0
        The width multiplier for controling the model size. The actual number of channels
        is equal to the original channel size multiplied by this multiplier.
    classes : int, default 1000
        Number of classes for the output layer.
    """

    def __init__(self, multiplier=1.0, classes=1000, **kwargs):
        super(MobileNetV2, self).__init__(**kwargs)
        self.features = nn.HybridSequential()
        _add_conv(self.features, int(32 * multiplier), kernel=3,
                  stride=2, pad=1, relu6=True)

        in_channels_group = [int(x * multiplier) for x in [32] + [16] + [24] * 2
                             + [32] * 3 + [64] * 4 + [96] * 3 + [160] * 3]
        channels_group = [int(x * multiplier) for x in [16] + [24] * 2 + [32] * 3
                          + [64] * 4 + [96] * 3 + [160] * 3 + [320]]
        ts = [1] + [6] * 16
        strides = [1, 2] * 2 + [1, 1, 2] + [1] * 6 + [2] + [1] * 3

        for in_c, c, t, s in zip(in_channels_group, channels_group, ts, strides):
            self.features.add(LinearBottleneck(in_channels=in_c, channels=c,
                                               t=t, stride=s))

        last_channels = int(1280 * multiplier) if multiplier > 1.0 else 1280
        _add_conv(self.features, last_channels, relu6=True)

        self.features.add(nn.GlobalAvgPool2D())

        self.output = nn.HybridSequential()
        self.output.add(
            nn.Conv2D(classes, 1, use_bias=False),
            nn.Flatten()
        )

    def forward(self, x):
        x = self.features(x)
        x = self.output(x)
        return x


# Constructor
@wrap_ctx_to_device_func
def get_mobilenet(multiplier, pretrained=False, device=cpu(),
                  root=os.path.join(base.data_dir(), 'models'), **kwargs):
    r"""MobileNet model from the
    `"MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications"
    <https://arxiv.org/abs/1704.04861>`_ paper.

    Parameters
    ----------
    multiplier : float
        The width multiplier for controling the model size. Only multipliers that are no
        less than 0.25 are supported. The actual number of channels is equal to the original
        channel size multiplied by this multiplier.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    device : Device, default CPU
        The device in which to load the pretrained weights.
    root : str, default $MXNET_HOME/models
        Location for keeping the model parameters.
    """
    net = MobileNet(multiplier, **kwargs)

    if pretrained:
        from ..model_store import get_model_file
        version_suffix = '{0:.2f}'.format(multiplier)
        if version_suffix in ('1.00', '0.50'):
            version_suffix = version_suffix[:-1]
        net.load_parameters(
            get_model_file(f'mobilenet{version_suffix}', root=root), device=device)
    return net


@wrap_ctx_to_device_func
def get_mobilenet_v2(multiplier, pretrained=False, device=cpu(),
                     root=os.path.join(base.data_dir(), 'models'), **kwargs):
    r"""MobileNetV2 model from the
    `"Inverted Residuals and Linear Bottlenecks:
    Mobile Networks for Classification, Detection and Segmentation"
    <https://arxiv.org/abs/1801.04381>`_ paper.

    Parameters
    ----------
    multiplier : float
        The width multiplier for controling the model size. Only multipliers that are no
        less than 0.25 are supported. The actual number of channels is equal to the original
        channel size multiplied by this multiplier.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    device : Device, default CPU
        The device in which to load the pretrained weights.
    root : str, default $MXNET_HOME/models
        Location for keeping the model parameters.
    """
    net = MobileNetV2(multiplier, **kwargs)

    if pretrained:
        from ..model_store import get_model_file
        version_suffix = '{0:.2f}'.format(multiplier)
        if version_suffix in ('1.00', '0.50'):
            version_suffix = version_suffix[:-1]
        net.load_parameters(
            get_model_file(f'mobilenetv2_{version_suffix}', root=root), device=device)
    return net


@wrap_ctx_to_device_func
def mobilenet1_0(**kwargs):
    r"""MobileNet model from the
    `"MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications"
    <https://arxiv.org/abs/1704.04861>`_ paper, with width multiplier 1.0.

    Parameters
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    device : Device, default CPU
        The device in which to load the pretrained weights.
    """
    return get_mobilenet(1.0, **kwargs)


@wrap_ctx_to_device_func
def mobilenet_v2_1_0(**kwargs):
    r"""MobileNetV2 model from the
    `"Inverted Residuals and Linear Bottlenecks:
    Mobile Networks for Classification, Detection and Segmentation"
    <https://arxiv.org/abs/1801.04381>`_ paper.

    Parameters
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    device : Device, default CPU
        The device in which to load the pretrained weights.
    """
    return get_mobilenet_v2(1.0, **kwargs)


@wrap_ctx_to_device_func
def mobilenet0_75(**kwargs):
    r"""MobileNet model from the
    `"MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications"
    <https://arxiv.org/abs/1704.04861>`_ paper, with width multiplier 0.75.

    Parameters
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    device : Device, default CPU
        The device in which to load the pretrained weights.
    """
    return get_mobilenet(0.75, **kwargs)


@wrap_ctx_to_device_func
def mobilenet_v2_0_75(**kwargs):
    r"""MobileNetV2 model from the
    `"Inverted Residuals and Linear Bottlenecks:
    Mobile Networks for Classification, Detection and Segmentation"
    <https://arxiv.org/abs/1801.04381>`_ paper.

    Parameters
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    device : Device, default CPU
        The device in which to load the pretrained weights.
    """
    return get_mobilenet_v2(0.75, **kwargs)


@wrap_ctx_to_device_func
def mobilenet0_5(**kwargs):
    r"""MobileNet model from the
    `"MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications"
    <https://arxiv.org/abs/1704.04861>`_ paper, with width multiplier 0.5.

    Parameters
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    device : Device, default CPU
        The device in which to load the pretrained weights.
    """
    return get_mobilenet(0.5, **kwargs)


@wrap_ctx_to_device_func
def mobilenet_v2_0_5(**kwargs):
    r"""MobileNetV2 model from the
    `"Inverted Residuals and Linear Bottlenecks:
    Mobile Networks for Classification, Detection and Segmentation"
    <https://arxiv.org/abs/1801.04381>`_ paper.

    Parameters
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    device : Device, default CPU
        The device in which to load the pretrained weights.
    """
    return get_mobilenet_v2(0.5, **kwargs)


@wrap_ctx_to_device_func
def mobilenet0_25(**kwargs):
    r"""MobileNet model from the
    `"MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications"
    <https://arxiv.org/abs/1704.04861>`_ paper, with width multiplier 0.25.

    Parameters
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    device : Device, default CPU
        The device in which to load the pretrained weights.
    """
    return get_mobilenet(0.25, **kwargs)


@wrap_ctx_to_device_func
def mobilenet_v2_0_25(**kwargs):
    r"""MobileNetV2 model from the
    `"Inverted Residuals and Linear Bottlenecks:
    Mobile Networks for Classification, Detection and Segmentation"
    <https://arxiv.org/abs/1801.04381>`_ paper.

    Parameters
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    device : Device, default CPU
        The device in which to load the pretrained weights.
    """
    return get_mobilenet_v2(0.25, **kwargs)
