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
"""Inception, implemented in Gluon."""
__all__ = ['Inception3', 'inception_v3']

import os

from ....context import cpu
from ...block import HybridBlock
from ... import nn
from ...contrib.nn import HybridConcurrent

# Helpers
def _make_basic_conv(**kwargs):
    out = nn.HybridSequential(prefix='')
    out.add(nn.Conv2D(use_bias=False, **kwargs))
    out.add(nn.BatchNorm(epsilon=0.001))
    out.add(nn.Activation('relu'))
    return out

def _make_branch(use_pool, *conv_settings):
    out = nn.HybridSequential(prefix='')
    if use_pool == 'avg':
        out.add(nn.AvgPool2D(pool_size=3, strides=1, padding=1))
    elif use_pool == 'max':
        out.add(nn.MaxPool2D(pool_size=3, strides=2))
    setting_names = ['channels', 'kernel_size', 'strides', 'padding']
    for setting in conv_settings:
        kwargs = {}
        for i, value in enumerate(setting):
            if value is not None:
                kwargs[setting_names[i]] = value
        out.add(_make_basic_conv(**kwargs))
    return out

def _make_A(pool_features, prefix):
    out = HybridConcurrent(axis=1, prefix=prefix)
    with out.name_scope():
        out.add(_make_branch(None,
                             (64, 1, None, None)))
        out.add(_make_branch(None,
                             (48, 1, None, None),
                             (64, 5, None, 2)))
        out.add(_make_branch(None,
                             (64, 1, None, None),
                             (96, 3, None, 1),
                             (96, 3, None, 1)))
        out.add(_make_branch('avg',
                             (pool_features, 1, None, None)))
    return out

def _make_B(prefix):
    out = HybridConcurrent(axis=1, prefix=prefix)
    with out.name_scope():
        out.add(_make_branch(None,
                             (384, 3, 2, None)))
        out.add(_make_branch(None,
                             (64, 1, None, None),
                             (96, 3, None, 1),
                             (96, 3, 2, None)))
        out.add(_make_branch('max'))
    return out

def _make_C(channels_7x7, prefix):
    out = HybridConcurrent(axis=1, prefix=prefix)
    with out.name_scope():
        out.add(_make_branch(None,
                             (192, 1, None, None)))
        out.add(_make_branch(None,
                             (channels_7x7, 1, None, None),
                             (channels_7x7, (1, 7), None, (0, 3)),
                             (192, (7, 1), None, (3, 0))))
        out.add(_make_branch(None,
                             (channels_7x7, 1, None, None),
                             (channels_7x7, (7, 1), None, (3, 0)),
                             (channels_7x7, (1, 7), None, (0, 3)),
                             (channels_7x7, (7, 1), None, (3, 0)),
                             (192, (1, 7), None, (0, 3))))
        out.add(_make_branch('avg',
                             (192, 1, None, None)))
    return out

def _make_D(prefix):
    out = HybridConcurrent(axis=1, prefix=prefix)
    with out.name_scope():
        out.add(_make_branch(None,
                             (192, 1, None, None),
                             (320, 3, 2, None)))
        out.add(_make_branch(None,
                             (192, 1, None, None),
                             (192, (1, 7), None, (0, 3)),
                             (192, (7, 1), None, (3, 0)),
                             (192, 3, 2, None)))
        out.add(_make_branch('max'))
    return out

def _make_E(prefix):
    out = HybridConcurrent(axis=1, prefix=prefix)
    with out.name_scope():
        out.add(_make_branch(None,
                             (320, 1, None, None)))

        branch_3x3 = nn.HybridSequential(prefix='')
        out.add(branch_3x3)
        branch_3x3.add(_make_branch(None,
                                    (384, 1, None, None)))
        branch_3x3_split = HybridConcurrent(axis=1, prefix='')
        branch_3x3_split.add(_make_branch(None,
                                          (384, (1, 3), None, (0, 1))))
        branch_3x3_split.add(_make_branch(None,
                                          (384, (3, 1), None, (1, 0))))
        branch_3x3.add(branch_3x3_split)

        branch_3x3dbl = nn.HybridSequential(prefix='')
        out.add(branch_3x3dbl)
        branch_3x3dbl.add(_make_branch(None,
                                       (448, 1, None, None),
                                       (384, 3, None, 1)))
        branch_3x3dbl_split = HybridConcurrent(axis=1, prefix='')
        branch_3x3dbl.add(branch_3x3dbl_split)
        branch_3x3dbl_split.add(_make_branch(None,
                                             (384, (1, 3), None, (0, 1))))
        branch_3x3dbl_split.add(_make_branch(None,
                                             (384, (3, 1), None, (1, 0))))

        out.add(_make_branch('avg',
                             (192, 1, None, None)))
    return out

def make_aux(classes):
    out = nn.HybridSequential(prefix='')
    out.add(nn.AvgPool2D(pool_size=5, strides=3))
    out.add(_make_basic_conv(channels=128, kernel_size=1))
    out.add(_make_basic_conv(channels=768, kernel_size=5))
    out.add(nn.Flatten())
    out.add(nn.Dense(classes))
    return out

# Net
class Inception3(HybridBlock):
    r"""Inception v3 model from
    `"Rethinking the Inception Architecture for Computer Vision"
    <http://arxiv.org/abs/1512.00567>`_ paper.

    Parameters
    ----------
    classes : int, default 1000
        Number of classification classes.
    """
    def __init__(self, classes=1000, **kwargs):
        super(Inception3, self).__init__(**kwargs)
        # self.use_aux_logits = use_aux_logits
        with self.name_scope():
            self.features = nn.HybridSequential(prefix='')
            self.features.add(_make_basic_conv(channels=32, kernel_size=3, strides=2))
            self.features.add(_make_basic_conv(channels=32, kernel_size=3))
            self.features.add(_make_basic_conv(channels=64, kernel_size=3, padding=1))
            self.features.add(nn.MaxPool2D(pool_size=3, strides=2))
            self.features.add(_make_basic_conv(channels=80, kernel_size=1))
            self.features.add(_make_basic_conv(channels=192, kernel_size=3))
            self.features.add(nn.MaxPool2D(pool_size=3, strides=2))
            self.features.add(_make_A(32, 'A1_'))
            self.features.add(_make_A(64, 'A2_'))
            self.features.add(_make_A(64, 'A3_'))
            self.features.add(_make_B('B_'))
            self.features.add(_make_C(128, 'C1_'))
            self.features.add(_make_C(160, 'C2_'))
            self.features.add(_make_C(160, 'C3_'))
            self.features.add(_make_C(192, 'C4_'))
            self.features.add(_make_D('D_'))
            self.features.add(_make_E('E1_'))
            self.features.add(_make_E('E2_'))
            self.features.add(nn.AvgPool2D(pool_size=8))
            self.features.add(nn.Dropout(0.5))

            self.output = nn.Dense(classes)

    def hybrid_forward(self, F, x):
        x = self.features(x)
        x = self.output(x)
        return x

# Constructor
def inception_v3(pretrained=False, ctx=cpu(),
                 root=os.path.join('~', '.mxnet', 'models'), **kwargs):
    r"""Inception v3 model from
    `"Rethinking the Inception Architecture for Computer Vision"
    <http://arxiv.org/abs/1512.00567>`_ paper.

    Parameters
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    net = Inception3(**kwargs)
    if pretrained:
        from ..model_store import get_model_file
        net.load_params(get_model_file('inceptionv3', root=root), ctx=ctx)
    return net
