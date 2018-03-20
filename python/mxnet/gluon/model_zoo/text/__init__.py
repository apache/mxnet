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
# pylint: disable=wildcard-import, arguments-differ
r"""Module for pre-defined NLP models.

This module contains definitions for the following model architectures:
-  `AWD`_

You can construct a model with random weights by calling its constructor:

.. code::

    from mxnet.gluon.model_zoo import text
    # TODO
    awd = text.awd_variant()

We provide pre-trained models for all the listed models.
These models can constructed by passing ``pretrained=True``:

.. code::

    from mxnet.gluon.model_zoo import text
    # TODO
    awd = text.awd_variant(pretrained=True)

.. _AWD: https://arxiv.org/abs/1404.5997
"""

from .base import *

from . import lm

def get_model(name, **kwargs):
    """Returns a pre-defined model by name

    Parameters
    ----------
    name : str
        Name of the model.
    pretrained : bool
        Whether to load the pretrained weights for model.
    classes : int
        Number of classes for the output layer.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.

    Returns
    -------
    HybridBlock
        The model.
    """
    #models = {'awd_variant': awd_variant}
    name = name.lower()
    if name not in models:
        raise ValueError(
            'Model %s is not supported. Available options are\n\t%s'%(
                name, '\n\t'.join(sorted(models.keys()))))
    return models[name](**kwargs)
