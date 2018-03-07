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
__all__ = ['Activation', 'LeakyReLU', 'PReLU', 'ELU', 'SELU', 'Swish']

from ... import initializer
from ..block import HybridBlock


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

    def hybrid_forward(self, F, x):
        return F.Activation(x, act_type=self._act_type, name='fwd')

    def __repr__(self):
        s = '{name}({_act_type})'
        return s.format(name=self.__class__.__name__,
                        **self.__dict__)


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

    def hybrid_forward(self, F, x):
        return F.LeakyReLU(x, act_type='leaky', slope=self._alpha, name='fwd')

    def __repr__(self):
        s = '{name}({alpha})'
        return s.format(name=self.__class__.__name__,
                        alpha=self._alpha)


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


    Inputs:
        - **data**: input tensor with arbitrary shape.

    Outputs:
        - **out**: output tensor with the same shape as `data`.
    """
    def __init__(self, alpha_initializer=initializer.Constant(0.25), **kwargs):
        super(PReLU, self).__init__(**kwargs)
        with self.name_scope():
            self.alpha = self.params.get('alpha', shape=(1,), init=alpha_initializer)

    def hybrid_forward(self, F, x, alpha):
        return F.LeakyReLU(x, gamma=alpha, act_type='prelu', name='fwd')


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

    def hybrid_forward(self, F, x):
        return F.where(x > 0, x, self._alpha * (F.exp(x) - 1.0))


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
        self._scale = 1.0507009873554804934193349852946
        self._alpha = 1.6732632423543772848170429916717

    def hybrid_forward(self, F, x):
        return self._scale * F.where(x > 0, x, self._alpha * (F.exp(x) - 1.0))


class Swish(HybridBlock):
    r"""
    Swish Activation function
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

    def hybrid_forward(self, F, x):
        return x * F.sigmoid(self._beta * x, name='fwd')
