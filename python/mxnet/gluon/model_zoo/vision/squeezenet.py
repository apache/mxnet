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
"""SqueezeNet, implemented in Gluon."""
__all__ = ['SqueezeNet', 'squeezenet1_0', 'squeezenet1_1']

import os

from ....context import cpu
from ...block import HybridBlock
from ... import nn
from ...contrib.nn import HybridConcurrent

# Helpers
def _make_fire(squeeze_channels, expand1x1_channels, expand3x3_channels):
    out = nn.HybridSequential(prefix='')
    out.add(_make_fire_conv(squeeze_channels, 1))

    paths = HybridConcurrent(axis=1, prefix='')
    paths.add(_make_fire_conv(expand1x1_channels, 1))
    paths.add(_make_fire_conv(expand3x3_channels, 3, 1))
    out.add(paths)

    return out

def _make_fire_conv(channels, kernel_size, padding=0):
    out = nn.HybridSequential(prefix='')
    out.add(nn.Conv2D(channels, kernel_size, padding=padding))
    out.add(nn.Activation('relu'))
    return out

# Net
class SqueezeNet(HybridBlock):
    r"""SqueezeNet model from the `"SqueezeNet: AlexNet-level accuracy with 50x fewer parameters
    and <0.5MB model size" <https://arxiv.org/abs/1602.07360>`_ paper.
    SqueezeNet 1.1 model from the `official SqueezeNet repo
    <https://github.com/DeepScale/SqueezeNet/tree/master/SqueezeNet_v1.1>`_.
    SqueezeNet 1.1 has 2.4x less computation and slightly fewer parameters
    than SqueezeNet 1.0, without sacrificing accuracy.

    Parameters
    ----------
    version : str
        Version of squeezenet. Options are '1.0', '1.1'.
    classes : int, default 1000
        Number of classification classes.
    """
    def __init__(self, version, classes=1000, **kwargs):
        super(SqueezeNet, self).__init__(**kwargs)
        assert version in ['1.0', '1.1'], ("Unsupported SqueezeNet version {version}:"
                                           "1.0 or 1.1 expected".format(version=version))
        with self.name_scope():
            self.features = nn.HybridSequential(prefix='')
            if version == '1.0':
                self.features.add(nn.Conv2D(96, kernel_size=7, strides=2))
                self.features.add(nn.Activation('relu'))
                self.features.add(nn.MaxPool2D(pool_size=3, strides=2, ceil_mode=True))
                self.features.add(_make_fire(16, 64, 64))
                self.features.add(_make_fire(16, 64, 64))
                self.features.add(_make_fire(32, 128, 128))
                self.features.add(nn.MaxPool2D(pool_size=3, strides=2, ceil_mode=True))
                self.features.add(_make_fire(32, 128, 128))
                self.features.add(_make_fire(48, 192, 192))
                self.features.add(_make_fire(48, 192, 192))
                self.features.add(_make_fire(64, 256, 256))
                self.features.add(nn.MaxPool2D(pool_size=3, strides=2, ceil_mode=True))
                self.features.add(_make_fire(64, 256, 256))
            else:
                self.features.add(nn.Conv2D(64, kernel_size=3, strides=2))
                self.features.add(nn.Activation('relu'))
                self.features.add(nn.MaxPool2D(pool_size=3, strides=2, ceil_mode=True))
                self.features.add(_make_fire(16, 64, 64))
                self.features.add(_make_fire(16, 64, 64))
                self.features.add(nn.MaxPool2D(pool_size=3, strides=2, ceil_mode=True))
                self.features.add(_make_fire(32, 128, 128))
                self.features.add(_make_fire(32, 128, 128))
                self.features.add(nn.MaxPool2D(pool_size=3, strides=2, ceil_mode=True))
                self.features.add(_make_fire(48, 192, 192))
                self.features.add(_make_fire(48, 192, 192))
                self.features.add(_make_fire(64, 256, 256))
                self.features.add(_make_fire(64, 256, 256))
            self.features.add(nn.Dropout(0.5))

            self.output = nn.HybridSequential(prefix='')
            self.output.add(nn.Conv2D(classes, kernel_size=1))
            self.output.add(nn.Activation('relu'))
            self.output.add(nn.AvgPool2D(13))
            self.output.add(nn.Flatten())

    def hybrid_forward(self, F, x):
        x = self.features(x)
        x = self.output(x)
        return x

# Constructor
def get_squeezenet(version, pretrained=False, ctx=cpu(),
                   root=os.path.join('~', '.mxnet', 'models'), **kwargs):
    r"""SqueezeNet model from the `"SqueezeNet: AlexNet-level accuracy with 50x fewer parameters
    and <0.5MB model size" <https://arxiv.org/abs/1602.07360>`_ paper.
    SqueezeNet 1.1 model from the `official SqueezeNet repo
    <https://github.com/DeepScale/SqueezeNet/tree/master/SqueezeNet_v1.1>`_.
    SqueezeNet 1.1 has 2.4x less computation and slightly fewer parameters
    than SqueezeNet 1.0, without sacrificing accuracy.

    Parameters
    ----------
    version : str
        Version of squeezenet. Options are '1.0', '1.1'.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    net = SqueezeNet(version, **kwargs)
    if pretrained:
        from ..model_store import get_model_file
        net.load_params(get_model_file('squeezenet%s'%version, root=root), ctx=ctx)
    return net

def squeezenet1_0(**kwargs):
    r"""SqueezeNet 1.0 model from the `"SqueezeNet: AlexNet-level accuracy with 50x fewer parameters
    and <0.5MB model size" <https://arxiv.org/abs/1602.07360>`_ paper.

    Parameters
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    return get_squeezenet('1.0', **kwargs)

def squeezenet1_1(**kwargs):
    r"""SqueezeNet 1.1 model from the `official SqueezeNet repo
    <https://github.com/DeepScale/SqueezeNet/tree/master/SqueezeNet_v1.1>`_.
    SqueezeNet 1.1 has 2.4x less computation and slightly fewer parameters
    than SqueezeNet 1.0, without sacrificing accuracy.

    Parameters
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    return get_squeezenet('1.1', **kwargs)
