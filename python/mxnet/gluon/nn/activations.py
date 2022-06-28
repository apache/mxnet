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
"""Basic neural network layers."""
__all__ = ['Activation', 'LeakyReLU', 'PReLU', 'ELU', 'SELU', 'Swish', 'GELU', 'SiLU']

from ... import initializer, npx
from ..block import HybridBlock
from ..parameter import Parameter
from ...util import use_np


@use_np
class Activation(HybridBlock):
    r"""Applies an activation function to input.

    Parameters
    ----------
    activation : str
        Name of activation function to use.
        See :func:`~mxnet.ndarray.Activation` for available choices.


    Inputs:
        - **data**: input tensor with arbitrary shape.

    Outputs:
        - **out**: output tensor with the same shape as `data`.
    """
    def __init__(self, activation, **kwargs):
        self._act_type = activation
        super(Activation, self).__init__(**kwargs)

    def _alias(self):
        return self._act_type

    def forward(self, x):
        return npx.activation(x, act_type=self._act_type, name='fwd')

    def __repr__(self):
        s = '{name}({_act_type})'
        return s.format(name=self.__class__.__name__,
                        **self.__dict__)


@use_np
class LeakyReLU(HybridBlock):
    r"""Leaky version of a Rectified Linear Unit.

    It allows a small gradient when the unit is not active

    .. math::

        f\left(x\right) = \left\{
            \begin{array}{lr}
               \alpha x & : x \lt 0 \\
                      x & : x \geq 0 \\
            \end{array}
        \right.\\

    Parameters
    ----------
    alpha : float
        slope coefficient for the negative half axis. Must be >= 0.


    Inputs:
        - **data**: input tensor with arbitrary shape.

    Outputs:
        - **out**: output tensor with the same shape as `data`.
    """
    def __init__(self, alpha, **kwargs):
        assert alpha >= 0, "Slope coefficient for LeakyReLU must be no less than 0."
        super(LeakyReLU, self).__init__(**kwargs)
        self._alpha = alpha

    def forward(self, x):
        return npx.leaky_relu(x, act_type='leaky', slope=self._alpha, name='fwd')

    def __repr__(self):
        s = '{name}({alpha})'
        return s.format(name=self.__class__.__name__,
                        alpha=self._alpha)


@use_np
class PReLU(HybridBlock):
    r"""Parametric leaky version of a Rectified Linear Unit.
    <https://arxiv.org/abs/1502.01852>`_ paper.

    It learns a gradient when the unit is not active

    .. math::

        f\left(x\right) = \left\{
            \begin{array}{lr}
               \alpha x & : x \lt 0 \\
                      x & : x \geq 0 \\
            \end{array}
        \right.\\

    where alpha is a learned parameter.

    Parameters
    ----------
    alpha_initializer : Initializer
        Initializer for the `embeddings` matrix.
    in_channels : int, default 1
        Number of channels (alpha parameters) to learn. Can either be 1
        or `n` where `n` is the size of the second dimension of the input
        tensor.

    Inputs:
        - **data**: input tensor with arbitrary shape.

    Outputs:
        - **out**: output tensor with the same shape as `data`.
    """
    def __init__(self, alpha_initializer=initializer.Constant(0.25),
                 in_channels=1, **kwargs):
        super(PReLU, self).__init__(**kwargs)
        self.alpha = Parameter('alpha', shape=(in_channels,), init=alpha_initializer)

    def forward(self, x):
        device = x.device
        return npx.leaky_relu(x, gamma=self.alpha.data(device), act_type='prelu', name='fwd')


@use_np
class ELU(HybridBlock):
    r"""
    Exponential Linear Unit (ELU)
        "Fast and Accurate Deep Network Learning by Exponential Linear Units", Clevert et al, 2016
        https://arxiv.org/abs/1511.07289
        Published as a conference paper at ICLR 2016

    Parameters
    ----------
    alpha : float
        The alpha parameter as described by Clevert et al, 2016


    Inputs:
        - **data**: input tensor with arbitrary shape.

    Outputs:
        - **out**: output tensor with the same shape as `data`.
    """

    def __init__(self, alpha=1.0, **kwargs):
        super(ELU, self).__init__(**kwargs)
        self._alpha = alpha

    def forward(self, x):
        return npx.leaky_relu(x, act_type='elu', slope=self._alpha)


@use_np
class SELU(HybridBlock):
    r"""
    Scaled Exponential Linear Unit (SELU)
        "Self-Normalizing Neural Networks", Klambauer et al, 2017
        https://arxiv.org/abs/1706.02515


    Inputs:
        - **data**: input tensor with arbitrary shape.

    Outputs:
        - **out**: output tensor with the same shape as `data`.
    """
    def __init__(self, **kwargs):
        super(SELU, self).__init__(**kwargs)

    def forward(self, x):
        return npx.leaky_relu(x, act_type='selu', name='fwd')


@use_np
class GELU(HybridBlock):
    r"""
    Gaussian Exponential Linear Unit (GELU)
        "Gaussian Error Linear Units (GELUs)", Hendrycks et al, 2016
        https://arxiv.org/abs/1606.08415

    Parameters
    ----------
    approximation : string
        Which approximation of GELU calculation to use (erf or tanh).

    Inputs:
        - **data**: input tensor with arbitrary shape.

    Outputs:
        - **out**: output tensor with the same shape as `data`.
    """
    def __init__(self, approximation='erf', **kwargs):
        if approximation not in ['erf', 'tanh']:
            raise ValueError("Unsupported approximation! Supported values are 'erf' and 'tanh', "
                             "but got '{}'".format(approximation))
        self._act_algorithm = 'gelu_' + approximation
        super(GELU, self).__init__(**kwargs)

    def forward(self, x):
        return npx.leaky_relu(x, act_type=self._act_algorithm, name='fwd')


@use_np
class Swish(HybridBlock):
    r"""
    Swish Activation function (SiLU with a hyperparameter)
        https://arxiv.org/pdf/1710.05941.pdf

    Parameters
    ----------
    beta : float
        swish(x) = x * sigmoid(beta*x)


    Inputs:
        - **data**: input tensor with arbitrary shape.

    Outputs:
        - **out**: output tensor with the same shape as `data`.
    """

    def __init__(self, beta=1.0, **kwargs):
        super(Swish, self).__init__(**kwargs)
        self._beta = beta

    def forward(self, x):
        return x * npx.sigmoid(self._beta * x)


@use_np
class SiLU(HybridBlock):
    r"""
    Sigmoid Linear Units
        Originally proposed "Gaussian Error Linear Units (GELUs)", Hendrycks et al, 2016
        https://arxiv.org/abs/1606.08415

    Parameters
    ----------
    beta : float
        silu(x) = x * sigmoid(x)


    Inputs:
        - **data**: input tensor with arbitrary shape.

    Outputs:
        - **out**: output tensor with the same shape as `data`.
    """

    def __init__(self, **kwargs):
        super(SiLU, self).__init__(**kwargs)

    def forward(self, x):
        return x * npx.sigmoid(x)
