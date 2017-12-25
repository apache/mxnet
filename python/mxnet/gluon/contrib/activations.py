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

# -*- coding: utf-8 -*-
''' This file contains definitions of advanced activation functions
for neural networks'''

from .. import gluon

class ELU(gluon.HybridBlock):
    r'''
    Exponential Linear Unit (ELU)
    ... "Fast and Accurate Deep Network Learning by Exponential Linear Units", Clevert et al, 2016
    ... https://arxiv.org/abs/1511.07289
    ... Published as a conference paper at ICLR 2016

    Parameters
    ----------
    alpha : float
        The alpha parameter as described by Clevert et al, 2016
    '''
    def __init__(self, alpha=1.0, **kwargs):
        super(ELU, self).__init__(**kwargs)
        self.alpha = alpha

    def hybrid_forward(self, F, x):
        return - self.alpha * F.relu(1.0 - F.exp(x)) + F.relu(x)


class SELU(gluon.HybridBlock):
    r'''
    Scaled Exponential Linear Unit (SELU)
    ... "Self-Normalizing Neural Networks", Klambauer et al, 2017
    ... https://arxiv.org/abs/1706.02515
    '''
    def __init__(self, **kwargs):
        super(SELU, self).__init__(**kwargs)
        self.scale = 1.0507009873554804934193349852946
        self.alpha = 1.6732632423543772848170429916717
        with self.name_scope():
            self.elu = ELU()

    def hybrid_forward(self, F, x):
        return self.scale * F.where(x >= 0, x, self.alpha * self.elu(x))


class Swish(gluon.HybridBlock):
    r'''
    Swish Activation function
    https://arxiv.org/pdf/1710.05941.pdf

    Parameters
    ----------
    beta : float
        swish(x) = x * sigmoid(beta*x)
    '''

    def __init__(self, beta=1.0, **kwargs):
        super(Swish, self).__init__(**kwargs)
        self.beta = beta

    def hybrid_forward(self, F, x):
        return x * F.Activation(self.beta * x, act_type='sigmoid', name='fwd')
