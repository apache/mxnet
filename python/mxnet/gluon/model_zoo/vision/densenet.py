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
"""DenseNet, implemented in Gluon."""
__all__ = ['DenseNet', 'densenet121', 'densenet161', 'densenet169', 'densenet201']

import os

from ....device import cpu
from ...block import HybridBlock
from ... import nn
from .... import base
from ....util import use_np, wrap_ctx_to_device_func

# Helpers
def _make_dense_block(num_layers, bn_size, growth_rate, dropout):
    out = nn.HybridSequential()
    for _ in range(num_layers):
        out.add(_make_dense_layer(growth_rate, bn_size, dropout))
    return out

def _make_dense_layer(growth_rate, bn_size, dropout):
    new_features = nn.HybridSequential()
    new_features.add(nn.BatchNorm())
    new_features.add(nn.Activation('relu'))
    new_features.add(nn.Conv2D(bn_size * growth_rate, kernel_size=1, use_bias=False))
    new_features.add(nn.BatchNorm())
    new_features.add(nn.Activation('relu'))
    new_features.add(nn.Conv2D(growth_rate, kernel_size=3, padding=1, use_bias=False))
    if dropout:
        new_features.add(nn.Dropout(dropout))

    out = nn.HybridConcatenate(axis=1)
    out.add(nn.Identity())
    out.add(new_features)

    return out

def _make_transition(num_output_features):
    out = nn.HybridSequential()
    out.add(nn.BatchNorm())
    out.add(nn.Activation('relu'))
    out.add(nn.Conv2D(num_output_features, kernel_size=1, use_bias=False))
    out.add(nn.AvgPool2D(pool_size=2, strides=2))
    return out

# Net
@use_np
class DenseNet(HybridBlock):
    r"""Densenet-BC model from the
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_ paper.

    Parameters
    ----------
    num_init_features : int
        Number of filters to learn in the first convolution layer.
    growth_rate : int
        Number of filters to add each layer (`k` in the paper).
    block_config : list of int
        List of integers for numbers of layers in each pooling block.
    bn_size : int, default 4
        Multiplicative factor for number of bottle neck layers.
        (i.e. bn_size * k features in the bottleneck layer)
    dropout : float, default 0
        Rate of dropout after each dense layer.
    classes : int, default 1000
        Number of classification classes.
    """
    def __init__(self, num_init_features, growth_rate, block_config,
                 bn_size=4, dropout=0, classes=1000, **kwargs):

        super(DenseNet, self).__init__(**kwargs)
        self.features = nn.HybridSequential()
        self.features.add(nn.Conv2D(num_init_features, kernel_size=7,
                                    strides=2, padding=3, use_bias=False))
        self.features.add(nn.BatchNorm())
        self.features.add(nn.Activation('relu'))
        self.features.add(nn.MaxPool2D(pool_size=3, strides=2, padding=1))
        # Add dense blocks
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            self.features.add(_make_dense_block(num_layers, bn_size, growth_rate, dropout))
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                self.features.add(_make_transition(num_features // 2))
                num_features = num_features // 2
        self.features.add(nn.BatchNorm())
        self.features.add(nn.Activation('relu'))
        self.features.add(nn.AvgPool2D(pool_size=7))
        self.features.add(nn.Flatten())

        self.output = nn.Dense(classes)

    def forward(self, x):
        x = self.features(x)
        x = self.output(x)
        return x


# Specification
densenet_spec = {121: (64, 32, [6, 12, 24, 16]),
                 161: (96, 48, [6, 12, 36, 24]),
                 169: (64, 32, [6, 12, 32, 32]),
                 201: (64, 32, [6, 12, 48, 32])}


# Constructor
@wrap_ctx_to_device_func
def get_densenet(num_layers, pretrained=False, device=cpu(),
                 root=os.path.join(base.data_dir(), 'models'), **kwargs):
    r"""Densenet-BC model from the
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_ paper.

    Parameters
    ----------
    num_layers : int
        Number of layers for the variant of densenet. Options are 121, 161, 169, 201.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    device : Device, default CPU
        The device in which to load the pretrained weights.
    root : str, default $MXNET_HOME/models
        Location for keeping the model parameters.
    """
    num_init_features, growth_rate, block_config = densenet_spec[num_layers]
    net = DenseNet(num_init_features, growth_rate, block_config, **kwargs)
    if pretrained:
        from ..model_store import get_model_file
        net.load_parameters(get_model_file(f'densenet{num_layers}', root=root), device=device)
    return net

def densenet121(**kwargs):
    r"""Densenet-BC 121-layer model from the
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_ paper.

    Parameters
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    device : Device, default CPU
        The device in which to load the pretrained weights.
    root : str, default '$MXNET_HOME/models'
        Location for keeping the model parameters.
    """
    return get_densenet(121, **kwargs)

def densenet161(**kwargs):
    r"""Densenet-BC 161-layer model from the
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_ paper.

    Parameters
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    device : Device, default CPU
        The device in which to load the pretrained weights.
    root : str, default '$MXNET_HOME/models'
        Location for keeping the model parameters.
    """
    return get_densenet(161, **kwargs)

def densenet169(**kwargs):
    r"""Densenet-BC 169-layer model from the
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_ paper.

    Parameters
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    device : Device, default CPU
        The device in which to load the pretrained weights.
    root : str, default '$MXNET_HOME/models'
        Location for keeping the model parameters.
    """
    return get_densenet(169, **kwargs)

def densenet201(**kwargs):
    r"""Densenet-BC 201-layer model from the
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_ paper.

    Parameters
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    device : Device, default CPU
        The device in which to load the pretrained weights.
    root : str, default '$MXNET_HOME/models'
        Location for keeping the model parameters.
    """
    return get_densenet(201, **kwargs)
