# -*- coding: utf-8 -*-
''' This file contains definitions of advanced activation functions
for neural networks'''

import mxnet.gluon as gluon
from mxnet import nd


class ELU(gluon.Block):
    '''
    Exponential Linear Unit (ELU)
    ... "Fast and Accurate Deep Network Learning by Exponential Linear Units"
    ... Djork-Arné Clevert, Thomas Unterthiner, Sepp Hochreiter
    ... https://arxiv.org/abs/1511.07289
    ... Published as a conference paper at ICLR 2016

    Parameters
    ----------
    alpha : float
        The alpha parameter as described by Clevert et Al 2016
    '''
    def __init__(self, alpha=1.0):
        super(ELU, self).__init__()
        self.alpha = alpha

    def forward(self, x):
        return - self.alpha * nd.relu(1.0 - nd.exp(x)) + nd.relu(x)


class SELU(gluon.Block):
    '''
    Scaled Exponential Linear Unit (SELU)
    ... "Self-Normalizing Neural Networks"
    ... Günter Klambauer, Thomas Unterthiner, Andreas Mayr, Sepp Hochreiter
    ... https://arxiv.org/abs/1706.02515
    '''
    def __init__(self):
        super(SELU, self).__init__()
        self.scale = 1.0507009873554804934193349852946
        self.alpha = 1.6732632423543772848170429916717
        with self.name_scope():
            self.elu = ELU()

    def forward(self, x):
        return self.scale * nd.where(x >= 0, x, self.alpha * self.elu(x))
